import json
import pandas as pd
import numpy as np


ADV_IDX_CONV = 'data/adv_idx_convert.csv'
ADV_IDX_CONV_TWO = 'data/adv_idx_convert_two_opt.json'

class AdvOptUnmatchedError(Exception):
    pass


class AdvIdxConverter():
    SLEEP = 'sleep'
    RESET = 'reset'
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
        for i in range(3):
            adv_type = self.conv_table.loc[adv_pred[i], 'type']
            if adv_type == 0:
                converted_adv_pred.append(
                    self.conv_table.loc[adv_pred[i], 'default']
                )

            elif adv_type == 1:
                possible_list = []
                for j in range(5):
                    possible_list.append(
                        self.conv_table.loc[adv_pred[i]-1, f'o_{j}']
                    )
                converted_adv_pred.append(possible_list)
            elif adv_type == 2:
                found = False
                for j, o in enumerate(options_idx):
                    if o == opt_one_pred[i]:
                        converted_adv_pred.append(
                            self.conv_table.loc[adv_pred[i], f'o_{j}']
                        )
                        found = True
                        break
                if not found:
                    raise AdvOptUnmatchedError(f'opt_one_pred {opt_one_pred[i]} is not in options_idx {options_idx}')

            elif adv_type == -1:
                found = False
                for j, o in enumerate(options_idx):
                    if o == opt_two_pred_1[i]:
                        opt_one_converted = j
                        found = True
                        break
                if not found:
                    raise AdvOptUnmatchedError(f'opt_two_pred_1 {opt_two_pred_1[i]} is not in options_idx {options_idx}')
                found = False
                for j, o in enumerate(options_idx):
                    if o == opt_two_pred_2[i]:
                        opt_two_converted = j
                        found = True
                        break
                if not found:
                    raise AdvOptUnmatchedError(f'opt_two_pred_2 {opt_two_pred_2[i]} is not in options_idx {options_idx}')
                converted_adv_pred.append(
                    self.conv_two_table[str(adv_pred[i])][f"{opt_one_converted}{opt_two_converted}"]
                )
            elif adv_type == -2:
                converted_adv_pred.append(
                    self.SLEEP
                )
            elif adv_type == -3:
                converted_adv_pred.append(
                    self.RESET
                )
            else:
                raise ValueError('adv_type is not in [-3, 2]')
        return converted_adv_pred