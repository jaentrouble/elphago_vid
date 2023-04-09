import numpy as np

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
        adv_sleeping: list[bool],
        enchant_n_pred: int,
        disable_left: int
    ):
        """get_value
        enchant_n_pred: Assumes enchant_n_pred is 0-indexed
        """
        tmp_opt = options.copy()
        for i, i_a in enumerate(is_avail):
            if not i_a:
                tmp_opt[i] = 0
        is_disable_turn = enchant_n_pred == disable_left
        disable_idx = disable_left-1 if is_disable_turn else disable_left
        adv_vals = []
        curve_vals = []
        for i, (a_s, a_p) in enumerate(zip(adv_sleeping, adv_pred)):
            adv_val = self.advice_table[
                tmp_opt[0],
                tmp_opt[1],
                tmp_opt[2],
                tmp_opt[3],
                tmp_opt[4],
                enchant_n_pred,
                a_p
            ]
            if enchant_n_pred > 1:
                tmp_adv_gauge = adv_gauge.copy()
                if a_s:
                    tmp_adv_gauge[i] = -7
                next_adv_gauge = self.next_adv_gauge(tmp_adv_gauge, i)
                curve_val = self.curve_table[
                    next_adv_gauge[0]+7,
                    next_adv_gauge[1]+7,
                    next_adv_gauge[2]+7,
                    enchant_n_pred-1,
                    disable_idx
                ]
            else:
                curve_val = None
            adv_vals.append(adv_val)
            curve_vals.append(curve_val)
        return adv_vals, curve_vals

    def adv_gauge_one_update(self, adv_gauge, chosen):
        if adv_gauge == -7:
            return
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