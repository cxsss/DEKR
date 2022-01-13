import torch
import torch.optim as optim
import argparse
import numpy as np
from model import DEKR
from data_loader import DataLoader, ModelDataset
from train import train
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# --------------------------------Parameter Setting--------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MachineLearning', help='which dataset to use')
parser.add_argument('--ratio', type=float, default=0.8, help='size of training set')
parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--dim', type=int, default=64, help='dimension of entity embeddings')
parser.add_argument('--desc_dim', type=float, default=300, help='original dimension of descriptive text embedding')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
args = parser.parse_args()

# --------------------------------Preparing dataset and KG--------------------------------
data_loader = DataLoader(args.dataset)
df_dataset = data_loader.load_dataset()
kg = data_loader.load_kg()
num_user, num_item, num_entity, num_relation = data_loader.get_num()
print('number of dataset:{}, number of method:{}'.format(num_user, num_item))

x_train, x_test, y_train, y_test = train_test_split(df_dataset, df_dataset['label'], test_size=1 - args.ratio, shuffle=True, random_state=2020)
train_dataset = ModelDataset(x_train)
test_dataset = ModelDataset(x_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
train_data = np.array(x_train)
test_data = np.array(x_test)


# --------------------------------Preparing the model--------------------------------
print('preparing model, loss function, optimizer...', end=' ')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DEKR(num_user, num_item, num_entity, num_relation, kg, args, device).to(device)
lossF = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
print('Done')
print('device: ', device)

# --------------------------------Training and Evaluating--------------------------------
print('training...')
show_topk = False
train(train_loader, test_loader, train_data, test_data, num_item, model, lossF, optimizer, device, args, show_topk)
