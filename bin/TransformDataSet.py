"""Time series transfrom
"""

import sys
import argparse 
from tqdm import tqdm
import numpy as np
sys.path.append('../CDPS/')
import CDPS_model as CDPS
import Utilities as utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='TransformDataSet',
                    description="Transform Data to the Embeding space. "
                                "The transfromation is done by using the"
                                " shapelets learnt by fitting a model."
                                " (i.e: A model that has leant shapelets"
                                " should be provided)"
                                )
    parser.add_argument(
                        'dataName',
                        help='Dataset Name',
                        type=str
                         )
    parser.add_argument(
                        'dataPath',
                        help='Dataset directory path',
                        type=str
                        )
    parser.add_argument(
                        'Type',
                        help="The scope of the Dataset "
                             "'TRAIN', 'TEST', or 'Alldata'",
                        type=str
                        )
    parser.add_argument(
                        'modelpath',
                        help='Path of the model to be loaded',
                        type=str
                        )
    parser.add_argument(
                        'Outdir',
                        help='Dirctory path to save the transformation',
                        type=str
                        )
    args = parser.parse_args()
    print(args)

    model = CDPS.CDPSModel.model_load(args.modelpath)
    if args.Type == "Aldata":
        Trdata, _ = utils.load_dataset_ts(args.dataName,
                                          'TRAIN',
                                          args.dataPath)
        Tsdata, _ = utils.load_dataset_ts(args.dataName,
                                          'TEST',
                                          args.dataPath)
        data = np.concatenate((Trdata, Tsdata))
        del Trdata, Tsdata
    else:
        data, _ = utils.load_dataset_ts(args.dataName,
                                        args.Type,
                                        args.dataPath)
    embTslearn = model._features(CDPS.tslearn2torch(data, model.device))
    embTslearnNumpy = embTslearn.cpu().detach().numpy()
    np.save(f'{args.Outdir}/{args.dataName}_Embeding_{args.Type}.model',
            embTslearnNumpy)
