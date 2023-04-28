import numpy as np
import index_converter

ADV_PATH = 'data/values/advice_counting_53_44_any_large2_re6_fl16.npz'
CURVE_PATH = 'data/values/curve_counting_53_44_any_v_re4.npy'

class ValueAnalyzer():
    def __init__(
        self,
        advice_path: str = ADV_PATH,
        curve_path: str = CURVE_PATH
    ):
        self.advice_table = np.load(advice_path)['data']
        self.curve_table = np.load(curve_path)

    def get_value(
        self,
        options: list[int],
        is_avail: list[bool],
        adv_gauge: list[int],
        adv_pred: list[int],
        enchant_n_pred: int,
    ):
        """get_value
        enchant_n_pred: Assumes enchant_n_pred is 0-indexed
        """
        tmp_opt = options.copy()
        for i, i_a in enumerate(is_avail):
            if not i_a:
                tmp_opt[i] = 11
        disable_left = np.sum(is_avail)-2
        is_disable_turn = enchant_n_pred == disable_left
        disable_idx = disable_left-1 if is_disable_turn else disable_left
        curve_vals = []
        tmp_adv_gauge = adv_gauge.copy()
        for i, a_p in enumerate(adv_pred):
            if a_p == index_converter.AdvIdxConverter.SLEEP:
                tmp_adv_gauge[i] = -7
        for i in range(3):
            if enchant_n_pred > 1:
                next_adv_gauge = self.next_adv_gauge(tmp_adv_gauge, i)
                curve_val = self.curve_table[
                    next_adv_gauge[0]+7,
                    next_adv_gauge[1]+7,
                    next_adv_gauge[2]+7,
                    enchant_n_pred-1,
                    disable_idx
                ]
            else:
                curve_val = 1
            curve_vals.append(curve_val)
        adv_vals = []
        final_vals = []
        max_final_vals = []
        for i, a_p in enumerate(adv_pred):
            # Unusual cases
            if isinstance(a_p, str):
                adv_vals.append(0)
                final_vals.append(0)
                max_final_vals.append(0)
                continue
            elif isinstance(a_p, list):
                adv_val = []
                final_val = []
                for a in a_p:
                    a = int(a)
                    adv_val.append(self.advice_table[
                        tmp_opt[0],
                        tmp_opt[1],
                        tmp_opt[2],
                        tmp_opt[3],
                        tmp_opt[4],
                        enchant_n_pred,
                        a
                    ])
                    final_val.append((adv_val[-1]**2)* curve_val)
                max_final_vals.append(max(final_val))
                
            else:
                a_p = int(a_p)
                adv_val = self.advice_table[
                    tmp_opt[0],
                    tmp_opt[1],
                    tmp_opt[2],
                    tmp_opt[3],
                    tmp_opt[4],
                    enchant_n_pred,
                    a_p
                ]
                final_val = (adv_val**2) * curve_val
                max_final_vals.append(final_val)
            adv_vals.append(adv_val)
            final_vals.append(final_val)
        return adv_vals, curve_vals, final_vals, np.argmax(max_final_vals)

    def adv_gauge_one_update(self, adv_gauge, chosen):
        if adv_gauge == -7:
            return -7
        if adv_gauge == 3 or adv_gauge == -6:
            adv_gauge = 0
        if chosen:
            adv_gauge = 1 if adv_gauge < 0 else adv_gauge + 1
        else:
            adv_gauge = -1 if adv_gauge > 0 else adv_gauge - 1
        return adv_gauge

    def next_adv_gauge(self, adv_gauge, advice_gauge_idx):
        new_adv_gauge = adv_gauge.copy()
        new_adv_gauge[advice_gauge_idx] = self.adv_gauge_one_update(
            adv_gauge[advice_gauge_idx], True)
        new_adv_gauge[(advice_gauge_idx+1)%3] = self.adv_gauge_one_update(
            adv_gauge[(advice_gauge_idx+1)%3], False)
        new_adv_gauge[(advice_gauge_idx+2)%3] = self.adv_gauge_one_update(
            adv_gauge[(advice_gauge_idx+2)%3], False)
        return new_adv_gauge