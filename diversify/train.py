import time
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ
from datautil.getdataloader_single import get_act_dataloader
from torch.utils.data import DataLoader

# ---- Automated K Estimation function ----
def automated_k_estimation(features, k_min=2, k_max=10):
    best_k = k_min
    best_score = -1
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(features)
        labels = kmeans.labels_
        score = silhouette_score(features, labels)
        if score > best_score:
            best_k = k
            best_score = score
    print(f"[INFO] Optimal K determined as {best_k} (Silhouette Score: {best_score:.4f})")
    return best_k

# ---- Warm-Up Pretraining Function ----
def warmup_pretraining(algorithm, train_loader, device, num_epochs=2, lr=1e-3):
    print(f"Starting warm-up pretraining for {num_epochs} epochs...")
    optimizer = torch.optim.Adam(algorithm.encoder.parameters(), lr=lr)
    algorithm.encoder.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in train_loader:
            x, y = batch[:2]
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            features = algorithm.encoder(x)
            logits = algorithm.clf(features)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Warm-up Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
    print("Warm-up pretraining done.")

def main(args):
    s = print_args(args, [])
    set_random_seed(args.seed)

    print_environ()
    print(s)
    train_loader, train_loader_noshuffle, valid_loader, target_loader, _, _, _ = get_act_dataloader(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).to(device)

    # ---- WARM-UP PRETRAINING ----
    warmup_pretraining(algorithm, train_loader, device, num_epochs=2, lr=args.lr)

    # ---- FEATURE EXTRACTION FOR K-ESTIMATION ----
    algorithm.eval()
    feature_list = []
    with torch.no_grad():
        for batch in train_loader:
            x = batch[0].to(device)
            features = algorithm.encoder(x)
            feature_list.append(features.cpu().numpy())
    all_features = np.concatenate(feature_list, axis=0)

    # ---- AUTOMATED K ESTIMATION ----
    optimal_k = automated_k_estimation(all_features)
    args.latent_domain_num = optimal_k
    print(f"Using automated latent_domain_num (K): {args.latent_domain_num}")

    # ---- (Optional) Update your dataset with new domain labels here, if needed ----
    # e.g., Assign kmeans.labels_ to samples, or update DataLoader if required by your algorithm

    # ---- Adjust batch size based on latent_domain_num ----
    if args.latent_domain_num < 6:
        args.batch_size = 32 * args.latent_domain_num
    else:
        args.batch_size = 16 * args.latent_domain_num

    best_valid_acc, target_acc = 0, 0

    # ---- MAIN DIVERSIFY TRAINING LOOP ----
    algorithm.train()
    optd = get_optimizer(algorithm, args, nettype='Diversify-adv')
    opt = get_optimizer(algorithm, args, nettype='Diversify-cls')
    opta = get_optimizer(algorithm, args, nettype='Diversify-all')

    for round in range(args.max_epoch):
        print(f'\n========ROUND {round}========')
        print('====Feature update====')
        loss_list = ['class']
        print_row(['epoch']+[item+'_loss' for item in loss_list], colwidth=15)

        for step in range(args.local_epoch):
            for data in train_loader:
                loss_result_dict = algorithm.update_a(data, opta)
            print_row([step]+[loss_result_dict[item] for item in loss_list], colwidth=15)

        print('====Latent domain characterization====')
        loss_list = ['total', 'dis', 'ent']
        print_row(['epoch']+[item+'_loss' for item in loss_list], colwidth=15)

        for step in range(args.local_epoch):
            for data in train_loader:
                loss_result_dict = algorithm.update_d(data, optd)
            print_row([step]+[loss_result_dict[item] for item in loss_list], colwidth=15)

        algorithm.set_dlabel(train_loader)

        print('====Domain-invariant feature learning====')
        loss_list = alg_loss_dict(args)
        eval_dict = train_valid_target_eval_names(args)
        print_key = ['epoch']
        print_key.extend([item+'_loss' for item in loss_list])
        print_key.extend([item+'_acc' for item in eval_dict.keys()])
        print_key.append('total_cost_time')
        print_row(print_key, colwidth=15)

        sss = time.time()
        for step in range(args.local_epoch):
            for data in train_loader:
                step_vals = algorithm.update(data, opt)

            results = {'epoch': step}
            results['train_acc'] = modelopera.accuracy(algorithm, train_loader_noshuffle, None)
            results['valid_acc'] = modelopera.accuracy(algorithm, valid_loader, None)
            results['target_acc'] = modelopera.accuracy(algorithm, target_loader, None)

            for key in loss_list:
                results[key+'_loss'] = step_vals[key]

            if results['valid_acc'] > best_valid_acc:
                best_valid_acc = results['valid_acc']
                target_acc = results['target_acc']
            results['total_cost_time'] = time.time()-sss
            print_row([results[key] for key in print_key], colwidth=15)

    print(f'Target acc: {target_acc:.4f}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # (your original arg definitions)
    args = parser.parse_args()
    main(args)
