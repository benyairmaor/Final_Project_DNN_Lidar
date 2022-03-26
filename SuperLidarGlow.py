import torch
import ETHDataSet as ETH
from torch.utils.data import DataLoader

################################################### Data Loader #################################################

# Load the ETH dataset
ETH_dataset = ETH.ETHDataset("apartment")

# print("=============== get item ==================")
# source, target, overlap, M = ETH_dataset[0]
# print(source.points, target.points, overlap, M)

# Split the data to train test
train_size = int(len(ETH_dataset) * 0.8)
test_size = len(ETH_dataset) - int(len(ETH_dataset) * 0.8)
train_set, test_set = torch.utils.data.random_split(ETH_dataset, [train_size, test_size])

# TrainLoader, 80% of the data
train_loader = DataLoader(
    train_set,
    batch_size=1,
    num_workers=0,
    shuffle=False
)

#  TestLoader, 20% of the data
test_loader = DataLoader(
    test_set,
    batch_size=1,
    num_workers=0,
    shuffle=False)

# # Display pcds and deatils for each problem.
# for batch_idx, (source, target, overlap, M) in enumerate(test_loader):
#     print("================", batch_idx, "==============")
#     print("source", source)
#     print("target", target)
#     print("overlap", overlap)
#     print("M", M)
