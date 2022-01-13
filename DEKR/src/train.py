import torch
from topK import topk_settings, topk_eval
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def train(train_loader, test_loader, train_data, test_data, num_item, model, lossF, optimizer, device, args, show_topk):
    loss_list = []
    auc_score_list = []
    if show_topk:
        print('preparing data for top-k recommendation...', end=' ')
        user_list, train_record, test_record, item_set, k_list = topk_settings(show_topk, train_data, test_data, num_item)
        print('Done')

    for epoch in range(args.n_epochs):
        running_loss = 0.0
        for i, (user_ids, item_ids, labels) in enumerate(train_loader):
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs_graph, outputs_desc = model(user_ids, item_ids)
            base_loss = lossF(outputs_graph, labels) + lossF(outputs_desc, labels)
            l2_loss = torch.tensor([0], dtype=torch.float32).to(device)
            for param in model.parameters():
                l2_loss += torch.norm(param, 2).to(device)
            loss = base_loss + args.l2_weight * l2_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        # print('[Epoch {}]train_loss: '.format(epoch + 1), running_loss / len(train_loader))
        loss_list.append(running_loss / len(train_loader))

        # --------------------------------Evaluating in CTR Prediction--------------------------------
        with torch.no_grad():
            test_loss = 0
            total_roc = 0
            total_acc = 0
            total_f1 = 0
            for user_ids, item_ids, labels in test_loader:
                user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
                outputs_graph, outputs_desc = model(user_ids, item_ids)
                outputs = (outputs_graph + outputs_desc) / 2
                outs = outputs.cpu().detach().numpy()
                predictions = [1 if i >= 0.5 else 0 for i in outs]

                test_loss += lossF(outputs, labels).item()
                total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                total_acc += accuracy_score(labels.cpu().detach().numpy(), predictions)
                total_f1 += f1_score(labels.cpu().detach().numpy(), predictions)
            # print(len(test_loader))
            print('[Epoch {}]test_loss: {} test_AUC: {} test_acc:{} test_f1: {}'.format((epoch + 1),
                                                                                        test_loss / len(test_loader),
                                                                                        total_roc / len(test_loader),
                                                                                        total_acc / len(test_loader),
                                                                                        total_f1 / len(test_loader)))
            # --------------------------------Evaluating in top-K Recommendation--------------------------------
            if show_topk:
                precision, recall, ndcg = topk_eval(model, user_list, train_record, test_record, item_set, k_list,
                                                    args.batch_size, device)
                print('precision: ', end='')
                for i in precision:
                    print('%.4f\t' % i, end='')
                print()
                print('recall: ', end='')
                for i in recall:
                    print('%.4f\t' % i, end='')
                print()
                print('ndcg: ', end='')
                for i in ndcg:
                    print('%.4f\t' % i, end='')
                print('\n')


