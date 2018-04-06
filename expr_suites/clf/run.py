"""
Implementation of few-shot adversarial domain adaptation - classification version
"""

import itertools
import os, sys
from ml_toolkit.pytorch_utils.misc import get_data_loader
from ml_toolkit.pytorch_utils.test_utils import load_models, save_test_results, run_classification_test
from ml_toolkit.pytorch_utils.train_utils import save_models, save_loss_records

sys.path.append(os.path.dirname(__file__))

from expr_suites.clf.train import train
from expr_suites.clf.pretrain import pretrain


def run_pretrain(model_def, params):
    "train encoder and hash generator on source data"
    save_model_to = "saved_models/pretrain"

    enc = model_def.LeNetEncoder()
    clf = model_def.LeNetClf(num_classes = params.num_classes)

    src_loader = get_data_loader(data_path=params.train_data_path["source"], dataset_mean=params.dataset_mean,
                                 dataset_std=params.dataset_std,
                                 batch_size=params.batch_size, shuffle_batch=True, image_scale=params.image_scale)

    train_results = pretrain(params=params, enc=enc, clf=clf, src_loader=src_loader)
    save_models(models=train_results["models"], save_model_to=save_model_to)
    for key, value in train_results["loss_records"].items():
        save_loss_records(loss_records=value, loss_name=key, save_to=save_model_to)


def main(model_def, params, pretrained_model_path=None):
    "training encoder, code generator using adversarial domain adaptation"
    save_model_to = "saved_models"

    # load models
    if (pretrained_model_path):
        pretrained_model = load_models(path=pretrained_model_path,model_names=["enc","clf"],test_mode=False)
        enc = pretrained_model["enc"]
        clf = pretrained_model["clf"]
        print("Pretrained enc and clf loaded")
    else:
        enc = model_def.LeNetEncoder()
        clf = model_def.LeNetClf(code_len=params.hash_size)

    dcd = model_def.Discriminator(input_dims=params.dcd_input_dims, hidden_dims=params.dcd_hidden_dims, output_dims=params.dcd_output_dims)

    # run training
    src_loader = get_data_loader(data_path=params.train_data_path["source"], dataset_mean=params.dataset_mean,
                    dataset_std=params.dataset_std,
                    batch_size=params.batch_size, shuffle_batch=True, image_scale=params.image_scale)
    tgt_loader = get_data_loader(data_path=params.train_data_path["target"], dataset_mean=params.dataset_mean,
                                 dataset_std=params.dataset_std,
                                 batch_size=params.batch_size, shuffle_batch=True, image_scale=params.image_scale)

    train_results = train(params=params,enc=enc,clf=clf,dcd=dcd,src_loader=src_loader,tgt_loader=tgt_loader)

    # save models and records
    save_models(models=train_results["models"],save_model_to=save_model_to)
    for key, value in train_results["loss_records"].items():
        save_loss_records(loss_records=value,loss_name=key,save_to=save_model_to)


def run_testing(params, model_path, model_def):
    # load data
    test_loader = get_data_loader(data_path=params.test_data_path,dataset_mean=params.dataset_mean,dataset_std=params.dataset_std,
                                   batch_size=params.batch_size,shuffle_batch=False,image_scale=params.image_scale)

    models = load_models(path=model_path,model_names=["enc","clf"],test_mode=True)
    enc, clf = models["enc"], models["clf"]
    clf_model = lambda x: clf(enc(x))
    result = run_classification_test(data_loader=test_loader,clf_model=clf_model) #TODO: this remains to be tested
    print(result)


########################################################################################
##############    The following are experiment suites to run  ##########################
########################################################################################

def suite_1():
    "pretraining on source"
    import model as model_def
    import params
    params.iterations = 300
    run_pretrain(model_def=model_def,params=params)

def suite_2():
    "adversarial domain adaptation training "
    import model as model_def
    import params
    main(model_def=model_def,params=params,pretrained_model_path="saved_models/pretrain")


def test_suite():
    import model as model_def
    import params
    # params.test_data_path = params.train_data_path["target"]
    run_testing(model_def=model_def, params=params, model_path="saved_models")


if __name__ == "__main__":
    test_suite()