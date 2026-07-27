# Identificação dos logs Gazebo RL

Os parâmetros abaixo foram identificados pelo conteúdo dos próprios logs:

- `tau_motor`: ajustado pela resposta discreta entre `rl_rpm_cmd` e
  `rl_motor_state`;
- noise: desvio entre `rl_obs_9..11` e `rate_p/q/r`;
- yaw invertido: sinal entre o desequilíbrio de torque dos rotores e
  `d(rate_r)/dt`.

| Log | tau_motor | Noise nas taxas | Yaw dos rotores | Observação |
| --- | ---: | --- | --- | --- |
| 20260723-074452 | 0.06 s | OFF | normal | primeira repetição da configuração-base |
| 20260723-075023 | 0.06 s | OFF | normal | segunda repetição da configuração-base |
| 20260723-133141 | 0.06 s | ON: 0.24/0.12/0.10 rad/s em p/q/r | normal | configuração com noise |
| 20260723-133626 | 0.06 s | ON: 0.24/0.12/0.10 rad/s em p/q/r | invertido | configuração com noise e yaw invertido |
| 20260723-143751 | 0.04 s | ON: 0.24/0.12/0.10 rad/s em p/q/r | invertido | o tau observado é 0.04 s, não 0.03 s |
| 20260723-144443 | 0.02 s | ON: 0.24/0.12/0.10 rad/s em p/q/r | invertido | configuração mais rápida dos motores |

## Cobertura das combinações com tau_motor = 0.06 s

| Noise | Yaw normal | Yaw invertido |
| --- | --- | --- |
| OFF | 20260723-074452 e 20260723-075023 | sem log encontrado |
| ON | 20260723-133141 | 20260723-133626 |

Não foi encontrado um log de `tau_motor=0.06 s`, noise OFF e yaw invertido.
