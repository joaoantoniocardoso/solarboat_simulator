Equações básicas:
* `batt_i = batt_i_out - batt_i_in` (convenção, eq.1)
* `esc_i_in = batt_i + mppts_i_out - oth_i_in = batt_i_out - batt_i_in + mppts_i_out - oth_i_in` (nó, eq.2)

A) de (2) se `esc_i_in == 0`:
```
esc_i_in = batt_i_out - batt_i_in + mppts_i_out - oth_i_in
0      = batt_i_out - batt_i_in + mppts_i_out - oth_i_in
batt_i_in = batt_i_out + mppts_i_out - oth_i_in
```

B) de (2) se `batt_i_out == 0`:
```
esc_i_in = batt_i_out - batt_i_in + mppts_i_out - oth_i_in
esc_i_in = 0       - batt_i_in + mppts_i_out - oth_i_in
esc_i_in = - batt_i_in + mppts_i_out - oth_i_in
```

C) de (2) se `duty_cycle > 0` tal que `motor_Z < 100 * batt_Z`, logo `batt_i_in ~= 0`, e então:
```
esc_i_in = batt_i_out - batt_i_in + mppts_i_out - oth_i_in
esc_i_in = batt_i_out - 0       + mppts_i_out - oth_i_in
esc_i_in = batt_i_out + mppts_i_out - oth_i_in
```

C.v2) de (2) se `duty_cycle > 0` tal que `esc_i_in >= mppts_i_out`, logo, `batt_i > 0` e consequentemente `batt_i_in == 0`, e então:
```
esc_i_in = batt_i_out - batt_i_in + mppts_i_out - oth_i_in
esc_i_in = batt_i_out - 0       + mppts_i_out - oth_i_in
esc_i_in = batt_i_out + mppts_i_out - oth_i_in
```

D) de (2) se `duty_cycle == 0`, logo `batt_i_out == 0` e `esc_i_in == 0`, e então:
```
esc_i_in = batt_i_out - batt_i_in + mppts_i_out - oth_i_in
0      = 0       - batt_i_in + mppts_i_out - oth_i_in
batt_i_in = mppts_i_out - oth_i_in
```

E) de (2) se `mppts_i_out == 0`, logo `batt_i_in == 0`, então:
```
esc_i_in = batt_i_out - batt_i_in + mppts_i_out - oth_i_in
esc_i_in = batt_i_out - 0       + 0        - oth_i_in
esc_i_in = batt_i_out - oth_i_in
```

F) de (1) se `batt_i > 0`, sabendo que ambos `batt_i_in` e `batt_i_out` pertencem aos `R >= 0`, então:
```
batt_i = batt_i_out - batt_i_in
batt_i_out > batt_i_in
```

G) de (1) se `batt_i < 0`, sabendo que ambos `batt_i_in` e `batt_i_out` pertencem aos `R >= 0`, então:
```
batt_i = batt_i_out - batt_i_in
batt_i_out < batt_i_in
```

H) de (2) se `duty_cycle > 0` tal que `esc_i_in < mppts_i_out`, logo, `batt_i < 0` e consequentemente `batt_i_out == 0`, e então:
```
esc_i_in = batt_i_out - batt_i_in + mppts_i_out - oth_i_in
esc_i_in = 0       - batt_i_in + mppts_i_out - oth_i_in
esc_i_in = - batt_i_in + mppts_i_out - oth_i_in
```

---
