import json
import pandas as pd
import numpy as np

ADV_IDX_CONV = 'data/adv_idx_convert.csv'
ADV_IDX_CONV_TWO = 'data/adv_idx_convert_two_opt.json'

class AdvIdxConverter():
    def __init__(self) -> None:
        self.conv_table = pd.read_csv(ADV_IDX_CONV)
        with open(ADV_IDX_CONV_TWO, 'r') as f:
            self.conv_two_table = json.load(f)

    def convert(
            self,
            options_idx: np.array,
            adv_pred: np.array,
            opt_one_pred: np.array,
            opt_two_pred_1: np.array,
            opt_two_pred_2: np.array
    ):
        """convert
        Convert 3 advices at one time
        Expected shapes:
            options_idx: (5,)
            adv_pred: (3,)
            opt_one_pred: (3,)
            opt_two_pred_1: (3,)
            opt_two_pred_2: (3,)
        """
        converted_adv_pred = []
        is_sleeping = [False]*3
        for i in range(3):
            adv_type = self.conv_table.loc[adv_pred[i], 'type']
            if adv_type == 0:
                for j, o in enumerate(options_idx):
                    if o == opt_one_pred[j]:
                        converted_one_opt = j
                        break
                    else:
                        raise ValueError('opt_one_pred is not in options_idx')
                converted_adv_pred.append(
                    self.conv_table.loc[adv_pred[i], f'o_{converted_one_opt}']
                )
            elif adv_type == 1:
                possible_advs = []
                for j in range(5):
                    possible_advs.append(
                        self.conv_table.loc[adv_pred[i]-1, f'o_{j}']
                    )
                    # TODO: Think of what to do with selectable advices (type 1)