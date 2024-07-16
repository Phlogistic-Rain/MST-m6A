import os
import torch
from sklearn import metrics
from torch.utils.data import DataLoader

import Util
from data_process import data_process, MyDataSet
from model import FusionDNAbert_1d_cov_2

batch_size = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def val(model, device, test_loader):
    model.eval()
    correct = 0

    total_num = len(test_loader.dataset)
    print('all val sample: ', total_num, ' / batch num: ', len(test_loader))
    y_pred = []
    y_true = []

    with torch.no_grad():
        for data, target in test_loader:
            target = target.to(device)
            output, out_layer = model(data)

            score_tmp = output
            _, pred = torch.max(output.data, 1)

            y_pred.extend(pred.view(-1).detach().cpu().numpy())
            y_true.extend(target.view(-1).detach().cpu().numpy())

            correct += torch.sum(pred == target)

        acc = (correct / total_num).item()
        pre = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred)
        mcc = metrics.matthews_corrcoef(y_true, y_pred)

        print('Acc:', acc)
        print('Pre:', pre)
        print('Sen:', recall)
        print('F1:', f1)
        print('Mcc:', mcc)


if __name__ == '__main__':

    cell_names = [
        'A549',
        'brain',
        'CD8T',
        'HCT116',
        'HEK293',
        'HEK293T',
        'HeLa',
        'HepG2',
        'kidney',
        'liver',
        'MOLM13'
    ]

    for cell_name in cell_names:
        path = './data/preprocessed_dataset'

        kmer1 = 4
        kmer2 = 6

        window_size = 201

        test_file = os.path.join(path, cell_name + '_test.tsv')
        test_data, test_label = data_process(test_file, window_size)

        torch.cuda.set_device(0)

        model = FusionDNAbert_1d_cov_2.FusionBERT(kmer1, kmer2)

        dict_path = './final_models/' + cell_name + '.pth'
        Util.load_state_dict(model, dict_path)

        model = model.to(device)

        print('Cell name: ', cell_name)

        test_dataset = MyDataSet(test_data, test_label, mutation=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        val(model, device, test_loader)
