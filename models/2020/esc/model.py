import control as ct
import numpy as np


class ESC:
    @classmethod
    def initial_state(cls, X0: dict, U0, params: dict):
        return np.array([])

    @classmethod
    def _outputs(cls, t, x, u, params: dict):
        esc_F_s = params["esc_F_s"]
        esc_V_ds_ov = params["esc_V_ds_ov"]
        esc_R_ds_on = params["esc_R_ds_on"]
        esc_E_on = params["esc_E_on"]
        esc_E_off = params["esc_E_off"]
        esc_V_F = params["esc_V_F"]
        esc_r_D = params["esc_r_D"]
        esc_Q_rr = params["esc_Q_rr"]

        esc_v_in = float(u[0])
        esc_i_out = float(u[1])
        esc_d = float(u[2])

        esc_vds_sw = esc_v_in * (1.0 + esc_V_ds_ov)
        esc_v_out = esc_d * esc_v_in

        if (esc_v_in <= 0.0) or (esc_d <= 0.0) or (esc_i_out <= 0.0):
            esc_p_m_c = 0.0
            esc_p_d_c = 0.0
            esc_p_m_sw = (
                (esc_E_on + esc_E_off) * esc_F_s * (esc_vds_sw / esc_v_in)
                if esc_v_in > 0
                else 0.0
            )
            esc_p_d_sw = esc_Q_rr * esc_vds_sw * esc_F_s
            esc_p_loss = esc_p_m_c + esc_p_d_c + esc_p_m_sw + esc_p_d_sw
            esc_i_in = (
                (esc_v_out * esc_i_out + esc_p_loss) / esc_v_in if esc_v_in > 0 else 0.0
            )
            return np.array([esc_v_out, esc_i_in])

        esc_i_m_rms = np.sqrt(esc_d) * esc_i_out
        esc_i_d_rms = np.sqrt(1.0 - esc_d) * esc_i_out
        esc_i_d_avg = (1.0 - esc_d) * esc_i_out

        esc_p_m_c = esc_i_m_rms**2 * esc_R_ds_on
        esc_p_d_c = esc_V_F * esc_i_d_avg + esc_r_D * esc_i_d_rms**2

        esc_p_m_sw = (esc_E_on + esc_E_off) * esc_F_s * (esc_vds_sw / esc_v_in)
        esc_p_d_sw = esc_Q_rr * esc_vds_sw * esc_F_s

        esc_p_loss = esc_p_m_c + esc_p_d_c + esc_p_m_sw + esc_p_d_sw

        esc_i_in = (esc_v_out * esc_i_out + esc_p_loss) / esc_v_in

        return np.array([esc_v_out, esc_i_in])

    @classmethod
    def build(cls, params: dict):
        return ct.NonlinearIOSystem(
            None,
            cls._outputs,
            name="esc",
            inputs=("esc_v_in", "esc_i_out", "esc_d"),
            outputs=("esc_v_out", "esc_i_in"),
            params=params,
        )
