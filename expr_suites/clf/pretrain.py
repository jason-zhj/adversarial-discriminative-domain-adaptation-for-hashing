"""
Pre-train encoder and code generator, using source data
"""

import torch
from ml_toolkit.data_process import make_variable
from ml_toolkit.pytorch_utils.loss import get_pairwise_sim_loss

def pretrain(enc, clf, src_loader, params):
    "enc is the encoder, h is code generator"
    opt_enc = torch.optim.Adam(enc.parameters(), lr=params.learning_rate)
    opt_clf = torch.optim.Adam(clf.parameters(), lr=params.learning_rate)

    loss_records = {"clf":[]}

    for i in range(params.iterations):

        loader = enumerate(src_loader)
        acc_loss = {key:0 for key in loss_records}

        for step, (images_src, labels_src) in loader:

            print("epoch {}/ batch {}".format(i,step))

            images_src = make_variable(images_src)
            g_src = enc(images_src)
            clf_src_out = clf(g_src)

            # cross-entropy loss
            criterion = torch.nn.CrossEntropyLoss()
            labels = torch.LongTensor(labels_src)
            clf_loss =  criterion(clf_src_out, make_variable(labels, requires_grad=False))

            acc_loss["clf"] += clf_loss.cpu().data.numpy()[0]

            clf_loss.backward()
            opt_enc.step(); opt_clf.step()
            opt_enc.zero_grad(); opt_clf.zero_grad()

        # record average loss
        for key in loss_records.keys():
            loss_records[key].append(acc_loss[key] / (step + 1))

    models = {
        "enc": enc,
        "clf": clf
    }
    return {
        "models": models,
        "loss_records": loss_records
    }