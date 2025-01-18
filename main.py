import torch.optim as optim
import sys
import math
from Models import *
from utility.helper import *
from utility.batch_test import *
from torch.optim.lr_scheduler import LambdaLR


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def lamb(epoch):
    epoch += 0
    return 0.95 ** (epoch / 14)

result = []
txt = open("./result.txt", "a")
alpha1=args.alpha1
alpha2=args.alpha2
def jaccard_similarity(matrix):
    intersection = np.dot(matrix, matrix.T)
    square_sum = np.diag(intersection)  # 获取对角线上的元素
    union = square_sum[:, None] + square_sum - intersection
    return np.divide(intersection, union)

class Model_Wrapper(object):
    def __init__(self, data_config, pretrain_data, warmup_epochs=5):
        
        # argument settings
        self.model_type = args.model_type
        self.weight_decay = 1e-2
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.mess_dropout = eval(args.mess_dropout)
        self.pretrain_data = pretrain_data
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.record_alphas = False
        self.lr = args.lr
        self.warmup_epochs = warmup_epochs  # Set warmup_epochs from the argument
        self.total_epochs = args.epoch
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.layer_num = args.layer_num
        self.model_type += '_%s_%s_layers%d' % (self.adj_type, self.alg_type, self.layer_num)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose

        print('model_type is {}'.format(self.model_type))

        self.weights_save_path = '%sweights/%s/%s/emb_size%s/layer_num%s/mess_dropout%s/drop_edge%s/lr%s/reg%s' % (
            args.weights_path, args.dataset, self.model_type,
            str(args.embed_size), str(args.layer_num), str(args.mess_dropout), str(args.drop_edge), str(args.lr),
            '-'.join([str(r) for r in eval(args.regs)]))
        self.result_message = []

        print('----self.alg_type is {}----'.format(self.alg_type))

        if self.alg_type in ['HyRec']:
            self.model = HyRec(self.n_users, self.n_items, self.emb_dim, self.layer_num, self.mess_dropout, 300)
        else:
            raise Exception('Dont know which model to train')

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Initialize the warmup scheduler and regular scheduler
        self.warmup_scheduler = self.set_warmup_scheduler()
        self.lr_scheduler = self.set_lr_scheduler()

        # Additional initialization code (tensor setup, etc.)
        self.adj_tensor_u1, self.adj_tensor_u2, self.adj_tensor_i1, self.adj_tensor_i2 = self.build_hyper_edge(
            args.data_path + args.dataset + '/TE.csv')

        self.model = self.model.cuda()
        self.adj_tensor_u1 = self.adj_tensor_u1.cuda()
        self.adj_tensor_u2 = self.adj_tensor_u2.cuda()
        self.adj_tensor_i1 = self.adj_tensor_i1.cuda()
        self.adj_tensor_i2 = self.adj_tensor_i2.cuda()
    
    
    def set_warmup_scheduler(self):
            # Lambda function for the warmup: Linear warmup until the base learning rate is reached
            def lr_lambda(epoch):
                if epoch < self.warmup_epochs:
                    return float(epoch) / float(max(1, self.warmup_epochs))
                else:
                    return 1.0  # After warmup, use the base learning rate
            
            # Using LambdaLR to apply the warmup logic
            return LambdaLR(self.optimizer, lr_lambda=lr_lambda)
    def get_D_inv(self, Hadj):

        H = sp.coo_matrix(Hadj.shape)
        H.row = Hadj.row.copy()
        H.col = Hadj.col.copy()
        H.data = Hadj.data.copy()
        rowsum = np.array(H.sum(1))
        columnsum = np.array(H.sum(0))

        Dv_inv = np.power(rowsum, -1).flatten()
        De_inv = np.power(columnsum, -1).flatten()
        Dv_inv[np.isinf(Dv_inv)] = 0.
        De_inv[np.isinf(De_inv)] = 0.

        Dv_mat_inv = sp.diags(Dv_inv)
        De_mat_inv = sp.diags(De_inv)
        return Dv_mat_inv, De_mat_inv

    def build_hyper_edge(self, file):
        user_inter = np.zeros((USR_NUM, ITEM_NUM))
        items_inter = np.zeros((ITEM_NUM, USR_NUM))
        
        with open(file) as f:
            for l in f.readlines():
                if len(l) == 0: break
                l = l.strip("\n").split(" ")
                uid = int(l[0])
                items = [int(j) for j in l[1:]]
                user_inter[uid, items] = 1
                items_inter[items, uid] = 1

        # Tensorized user interaction matrix
        J_u = jaccard_similarity(user_inter)
        indices_u = np.where(J_u > alpha1)
        values_u = J_u[indices_u]
        
        # Tensor for user adjacency (adj_tensor_u1)
        adj_indices_u1 = np.vstack([indices_u[0], indices_u[1], np.zeros_like(indices_u[0])])  # Tensor index array
        adj_values_u1 = values_u
        adj_tensor_u1 = torch.sparse_coo_tensor(adj_indices_u1, adj_values_u1, torch.Size([USR_NUM, USR_NUM, USR_NUM]))  # 3D tensor

        # Transposed tensor for users (adj_tensor_u2)
        adj_indices_u2 = np.vstack([indices_u[1], indices_u[0], np.zeros_like(indices_u[0])])  # Transpose the tensor indices
        adj_values_u2 = values_u
        adj_tensor_u2 = torch.sparse_coo_tensor(adj_indices_u2, adj_values_u2, torch.Size([USR_NUM, USR_NUM, USR_NUM]))  # 3D tensor
        
        # Tensorized item interaction matrix
        J_i = jaccard_similarity(items_inter)
        indices_i = np.where(J_i > alpha2)
        values_i = J_i[indices_i]

        # Tensor for item adjacency (adj_tensor_i1)
        adj_indices_i1 = np.vstack([indices_i[0], indices_i[1], np.zeros_like(indices_i[0])])
        adj_values_i1 = values_i
        adj_tensor_i1 = torch.sparse_coo_tensor(adj_indices_i1, adj_values_i1, torch.Size([ITEM_NUM, ITEM_NUM, ITEM_NUM]))

        # Transposed tensor for items (adj_tensor_i2)
        adj_indices_i2 = np.vstack([indices_i[1], indices_i[0], np.zeros_like(indices_i[0])])
        adj_values_i2 = values_i
        adj_tensor_i2 = torch.sparse_coo_tensor(adj_indices_i2, adj_values_i2, torch.Size([ITEM_NUM, ITEM_NUM, ITEM_NUM]))

        return adj_tensor_u1, adj_tensor_u2, adj_tensor_i1, adj_tensor_i2

    def set_lr_scheduler(self):  # lr_scheduler：学习率调度器
        fac = lamb
        scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 80], gamma=0.1)
        return scheduler

    def save_model(self):
        ensureDir(self.weights_save_path)
        torch.save(self.model.state_dict(), self.weights_save_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.weights_save_path))

    def test(self, users_to_test, drop_flag=False, batch_test_flag=False):
        self.model.eval()  # 评估模式，batchnorm和Drop层不起作用，相当于self.model.train(False)
        with torch.no_grad():
            ua_embeddings, ia_embeddings = self.model(self.adj_tensor_u1, self.adj_tensor_u2, self.adj_tensor_i1, self.adj_tensor_i2)
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test)
        return result

    def train(self):
        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger, map_loger, mrr_loger, fone_loger = [], [], [], [], [], [], [], []
        stopping_step = 20
        should_stop = False
        cur_best_pre_0 = 0.0
        n_batch = data_generator.n_train // args.batch_size + 1
        
        for epoch in range(args.epoch):
            t1 = time()
            epoch_loss, epoch_mf_loss, epoch_emb_loss, epoch_reg_loss = 0.0, 0.0, 0.0, 0.0
            sample_time = 0.0

            for idx in range(n_batch):
                self.model.train()  # Set the model to training mode
                self.optimizer.zero_grad()

                # Sample batch of users, positive items, and negative items
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()
                sample_time += time() - sample_t1

                # Get embeddings
                ua_embeddings, ia_embeddings = self.model(self.adj_tensor_u1, self.adj_tensor_u2, self.adj_tensor_i1, self.adj_tensor_i2)

                u_g_embeddings = ua_embeddings[users]
                pos_i_g_embeddings = ia_embeddings[pos_items]
                neg_i_g_embeddings = ia_embeddings[neg_items]

                # Pass epoch to hinge_loss
                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.parewise_hinge_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
                #batch_mf_loss, batch_emb_loss, batch_reg_loss = self.parewise_hinge_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings, epoch)

                batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss
                batch_loss.backward()
                self.optimizer.step()

                # Accumulate loss metrics for the epoch
                epoch_loss += float(batch_loss)
                epoch_mf_loss += float(batch_mf_loss)
                epoch_emb_loss += float(batch_emb_loss)
                epoch_reg_loss += float(batch_reg_loss)

            if epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.lr_scheduler.step()  # Regular scheduler after warmup

            self.lr_scheduler.step(epoch)  # Update learning rate

            # Ensure embeddings are not holding unneeded memory
            del ua_embeddings, ia_embeddings, u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings

            # Check if loss is NaN
            if math.isnan(epoch_loss):
                print('ERROR: loss is nan.')
                sys.exit()

            # Print and log performance every 10 epochs
            if (epoch + 1) % 10 == 0:
                perf_str = f'Epoch {epoch} [{time() - t1:.1f}s]: train==[{epoch_loss:.5f}={epoch_mf_loss:.5f} + {epoch_emb_loss:.5f}]'
                print(perf_str)

            # Evaluation on test set
            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            ret = self.test(users_to_test, drop_flag=True)
            training_time_list.append(t2 - t1)

            # Log results
            loss_loger.append(epoch_loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])
            map_loger.append(ret['map'])
            mrr_loger.append(ret['mrr'])
            fone_loger.append(ret['fone'])

            if args.verbose > 0:
                # Update perf_str to include k=20 results along with k=5 and k=10
                perf_str = (
                    f'Epoch {epoch} [{time() - t1:.1f}s + {t2 - t1:.1f}s]: '
                    f'train==[{epoch_loss:.5f}={epoch_mf_loss:.5f} + {epoch_emb_loss:.5f} + {epoch_reg_loss:.5f}], '
                    f'recall=[{ret["recall"][0]:.5f}, {ret["recall"][1]:.5f}, {ret["recall"][2]:.5f}], '  # For k=5, k=10, k=20
                    f'precision=[{ret["precision"][0]:.5f}, {ret["precision"][1]:.5f}, {ret["precision"][2]:.5f}], '  # For k=5, k=10, k=20
                    f'hit=[{ret["hit_ratio"][0]:.5f}, {ret["hit_ratio"][1]:.5f}, {ret["hit_ratio"][2]:.5f}], '  # For k=5, k=10, k=20
                    f'ndcg=[{ret["ndcg"][0]:.5f}, {ret["ndcg"][1]:.5f}, {ret["ndcg"][2]:.5f}], '  # For k=5, k=10, k=20
                    f'map=[{ret["map"][0]:.5f}, {ret["map"][1]:.5f}, {ret["map"][2]:.5f}], '  # For k=5, k=10, k=20
                    f'mrr=[{ret["mrr"][0]:.5f}, {ret["mrr"][1]:.5f}, {ret["mrr"][2]:.5f}], '  # For k=5, k=10, k=20
                    f'f1=[{ret["fone"][0]:.5f}, {ret["fone"][1]:.5f}, {ret["fone"][2]:.5f}]'  # For k=5, k=10, k=20
                )

                result.append(perf_str + "\n")
                global txt
                txt.write(perf_str + "\n")
                txt.close()
                txt = open("./result.txt", "a")
                print(perf_str)

            # Early stopping based on recall at index 0
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0, stopping_step, expected_order='acc', flag_step=150)
            if should_stop:
                break

            # Save the model if the best performance is achieved
            if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
                self.save_model()
                if self.record_alphas:
                    self.best_alphas = [i for i in self.model.get_alphas()]
                print(f'Saving the model at epoch {epoch} to path: {self.weights_save_path}')

        # Save final recommendation results to CSV
        if args.save_recom:
            results_save_path = f'./output/{args.dataset}/rec_result.csv'
            self.save_recResult(results_save_path)

        # Print final results
        if rec_loger:
            self.print_final_results(rec_loger, pre_loger, ndcg_loger, hit_loger, map_loger, mrr_loger, fone_loger, training_time_list)



    def norm(self, adj):
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    def save_recResult(self, outputPath):
        # used for reverve the recommendation lists
        recommendResult = {}
        u_batch_size = BATCH_SIZE * 2
        i_batch_size = BATCH_SIZE

        # get all apps (users)
        users_to_test = list(data_generator.test_set.keys())
        n_test_users = len(users_to_test)
        n_user_batchs = n_test_users // u_batch_size + 1
        count = 0

        # calculate the result by our own
        # get the latent factors
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings = self.model(self.adj_tensor_u1, self.adj_tensor_u2, self.adj_tensor_i1,
                                                      self.adj_tensor_i2)

        # get result in batch
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            user_batch = users_to_test[start: end]
            item_batch = range(ITEM_NUM)
            u_g_embeddings = ua_embeddings[user_batch]
            i_g_embeddings = ia_embeddings[item_batch]
            # get the ratings
            rate_batch = torch.matmul(u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1))
            # move from GPU to CPU
            rate_batch = rate_batch.detach().cpu().numpy()
            # contact each user's ratings with his id
            user_rating_uid = zip(rate_batch, user_batch)
            # now for each user, calculate his ratings and recommendation
            for x in user_rating_uid:
                # user u's ratings for user u
                rating = x[0]
                # uid
                u = x[1]
                training_items = data_generator.train_items[u]
                user_pos_test = data_generator.test_set[u]
                all_items = set(range(ITEM_NUM))
                test_items = list(all_items - set(training_items))
                item_score = {}
                for i in test_items:
                    item_score[i] = rating[i]
                K_max = max(Ks)
                K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
                recommendResult[u] = K_max_item_score

        # output the result to csv file.
        ensureDir(outputPath)
        with open(outputPath, 'w') as f:
            print("----the recommend result has %s items." % (len(recommendResult)))
            for key in recommendResult.keys():  # due to that all users have been used for test and the subscripts start from 0.
                outString = ""
                for v in recommendResult[key]:
                    outString = outString + "," + str(v)
                f.write("%s%s\n" % (key, outString))

    def print_final_results(self, rec_loger, pre_loger, ndcg_loger, hit_loger, map_loger, mrr_loger, fone_loger,
                            training_time_list):
        recs = np.array(rec_loger)
        pres = np.array(pre_loger)
        map = np.array(map_loger)
        mrr = np.array(mrr_loger)
        fone = np.array(fone_loger)

        best_rec_0 = max(recs[:, 0])
        idx = list(recs[:, 0]).index(best_rec_0)

        final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s], map=[%s],mrr=[%s], f1=[%s]" % \
                     (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                      '\t'.join(['%.5f' % r for r in pres[idx]]),
                      '\t'.join(['%.5f' % r for r in hit_loger[idx]]),
                      '\t'.join(['%.5f' % r for r in ndcg_loger[idx]]),
                      '\t'.join(['%.5f' % r for r in map[idx]]),
                      '\t'.join(['%.5f' % r for r in mrr[idx]]),
                      '\t'.join(['%.5f' % r for r in fone[idx]]))
        result.append(final_perf + "\n")
        txt.write(final_perf + "\n")
        print(final_perf)

    # def bpr_loss(self, users, pos_items, neg_items):
    #     pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)  # torch.mul():对应元素相乘
    #     neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)  # torch.mul():对应元素相乘

    #     regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
    #     regularizer = regularizer / self.batch_size

    #     maxi = F.logsigmoid(pos_scores - neg_scores)
    #     mf_loss = -torch.mean(maxi)

    #     emb_loss = self.decay * regularizer
    #     reg_loss = 0.0
    #     return mf_loss, emb_loss, reg_loss


    # def parewise_hinge_loss(self, users, pos_items, neg_items, epoch, margin=1.5):
    #     """
    #     Pairwise Hinge loss implementation for ranking.
        
    #     Args:
    #     - users: User embeddings.
    #     - pos_items: Positive item embeddings (for positive interactions).
    #     - neg_items: Negative item embeddings (for negative interactions).
    #     - epoch: Current epoch number.
    #     - margin: The margin for hinge loss (default is 1.0).
        
    #     Returns:
    #     - mf_loss: Margin-based ranking loss.
    #     - emb_loss: Regularization loss for embeddings.
    #     - reg_loss: Weight regularization term.
    #     """
        
    #     # Compute positive and negative scores using dot products (or element-wise multiplication)
    #     pos_scores = torch.sum(users * pos_items, dim=1)  # Positive interaction scores
    #     neg_scores = torch.sum(users * neg_items, dim=1)  # Negative interaction scores

    #     # Hinge loss formula: max(0, margin - (positive_score - negative_score))
    #     hinge_losses = torch.clamp(margin - (pos_scores - neg_scores), min=0)

    #     # Mean hinge loss
    #     mf_loss = torch.mean(hinge_losses)

    #     # Embedding regularization term (to prevent overfitting)
    #     regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
    #     emb_loss = self.decay * regularizer / self.batch_size

    #     # L2 regularization (weight decay)
    #     reg_loss = self.weight_decay * (self.model.user_embedding.weight.norm(2) ** 2 + self.model.item_embedding.weight.norm(2) ** 2)
        
    #     return mf_loss, emb_loss, reg_loss

    # def parewise_hinge_loss(self, users, pos_items, neg_items):
    #     """
    #         Bayesian Personalized Ranking (BPR) loss for ranking.
    #     """
    #     # Compute positive and negative scores
    #     pos_scores = torch.sum(users * pos_items, dim=1)  # Positive interaction scores
    #     neg_scores = torch.sum(users * neg_items, dim=1)  # Negative interaction scores
        
    #     # BPR loss: log-sigmoid of the difference between positive and negative scores
    #     bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
        
    #     # Regularization
    #     regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
    #     emb_loss = self.decay * regularizer / self.batch_size
        
    #     # L2 regularization (weight decay)
    #     reg_loss = self.weight_decay * (self.model.user_embedding.weight.norm(2) ** 2 + self.model.item_embedding.weight.norm(2) ** 2)
        
    #     return bpr_loss, emb_loss, reg_loss
    #    #Best Iter=[260]@[297.1] recall=[0.30722 0.40512 0.51530], precision=[0.28527    0.19338 0.12690], hit=[0.815990.88671 0.93671], ndcg=[0.65460 0.659370.64665], map=[0.58351   0.54691 0.48979],mrr=[0.62000   0.62987 0.63336], f1=[0.27862   0.24596 0.19260]

    def parewise_hinge_loss(self, users, pos_items, neg_items, margin=1.0):
        """
        Max-Margin Ranking Loss for ranking.
        """
        # Compute positive and negative scores
        pos_scores = torch.sum(users * pos_items, dim=1)  # Positive interaction scores
        neg_scores = torch.sum(users * neg_items, dim=1)  # Negative interaction scores
        
        # Max-margin loss
        max_margin_loss = torch.mean(torch.clamp(margin - (pos_scores - neg_scores), min=0))
        
        # Regularization
        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        emb_loss = self.decay * regularizer / self.batch_size
        
        # L2 regularization (weight decay)
        reg_loss = self.weight_decay * (self.model.user_embedding.weight.norm(2) ** 2 + self.model.item_embedding.weight.norm(2) ** 2)
        
        return max_margin_loss, emb_loss, reg_loss
    #Best Iter=[116]@[198.5] recall=[0.31041 0.41641 0.52149], precision=[0.28833    0.19827 0.12818], hit=[0.81509  0.88784 0.93964], ndcg=[0.65464 0.658000.64804], map=[0.58398   0.54316 0.49095],mrr=[0.61858   0.62851 0.63216], f1=[0.28170   0.25239 0.19454]
    
    def parewisde_hinge_loss(self, users, pos_items, neg_items, temperature=0.5):
        # Compute scores
        # Compute scores
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)
        
        # Apply temperature scaling and softmax
        scaled_pos_scores = pos_scores / temperature
        scaled_neg_scores = neg_scores / temperature
        
        # Softmax loss with temperature scaling
        softmax_loss = torch.mean(-torch.log(torch.exp(scaled_pos_scores) / (torch.exp(scaled_pos_scores) + torch.exp(scaled_neg_scores))))
        
        # Regularization terms (unchanged)
        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        emb_loss = self.decay * regularizer / self.batch_size
        reg_loss = self.weight_decay * (self.model.user_embedding.weight.norm(2) ** 2 + self.model.item_embedding.weight.norm(2) ** 2)
        
        return softmax_loss, emb_loss, reg_loss

    


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    t0 = time()

    pretrain_data = None

    # Pass the warmup_epochs from args to Model_Wrapper
    Engine = Model_Wrapper(data_config=config, pretrain_data=pretrain_data, warmup_epochs=args.warmup_epochs)

    if args.pretrain:
        print('pretrain path: ', Engine.weights_save_path)
        if os.path.exists(Engine.weights_save_path):
            Engine.load_model()
            users_to_test = list(data_generator.test_set.keys())
            ret = Engine.test(users_to_test, drop_flag=True)
            cur_best_pre_0 = ret['recall'][0]

            pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                           'ndcg=[%.5f, %.5f], map=[%.5f, %.5f], mrr=[%.5f, %.5f], f1=[%.5f, %.5f]' % \
                           (ret['recall'][0], ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1],
                            ret['hit_ratio'][0], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1],
                            ret['map'][0], ret['map'][-1],
                            ret['mrr'][0], ret['mrr'][-1],
                            ret['fone'][0], ret['fone'][-1])
            print(pretrain_ret)
        else:
            print('Cannot load pretrained model. Start training from scratch')
    else:
        print('without pretraining')
    
    Engine.train()





