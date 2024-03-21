# TODO:

## Needed
    > [Done] Use hypothesis
    > [Done] Fix battery
    > [Done] Add multiple boats in the simulation
    > [Done] Convert results to DataFrame
    > Implement and use boat status
    > [WIP] Save/load results
    > [WIP] Graphs / results presentation

## Nice to have:
    > Use loguru
    > Use tqdm


\rm -rf /home/joaoantoniocardoso/ZeniteSolar/2022/Strategy22/datasets/can/parsed/sparse/*
\rm -rf /home/joaoantoniocardoso/ZeniteSolar/2022/Strategy22/datasets/can/final/*
\rm -rf /home/joaoantoniocardoso/ZeniteSolar/2022/Strategy22/datasets/can/parsed/1s/*
\rm -rf /home/joaoantoniocardoso/ZeniteSolar/2022/Strategy22/datasets/nonideal_solar_dataset.csv


Tendo:
    - o dado de controle do piloto das provas de 2019 ou 2020,
    - os tempos reais das provas,
    - tendo os dados climáticos durante as competições,
    - tendo conhecimento sobre os componentes internos da embarcação
Então, se o modelo proposto consegue representar a performance da embarcação, será possível parametrizar o modelo utilizando os dados de uma prova, e utilizar o mesmo modelo para prever o resultado de outras provas.


A. Estimar eficiência de saída casco
    1. Estimar a distância das provas -> [OK]
        a. Distância de cada volta
        b. Número de voltas
    2. Estimar o tempo real da embarcação nas provas -> [OK]
        a. Com base nos resultados da competição
        b. Com base nos dados da rede CAN
    3. Calcular a velocidade média nas provas -> [OK]
        a. (distância total na prova) / (tempo real da embarcação nas provas)
    5. Calcular o duty-cycle médio do controlador do motor nas provas -> [OK]
    6. Eff_out = Eff_esc * Eff_motor * Eff_transmissao * Eff_casco
        a. Estimar Eff_esc com base em ensaios realizados -> [OK]
        b. Estimar Eff_motor com baase nas curvas do fabricante -> [OK]
        c. Estimar Eff_transmissao -> [OK] 80% (Dudlle 1954 diz que uma bevel gears teria por volta de 98%, "Efficiency measurement campaign on gearboxes" mostra como pode ser diferente do especificado pelo catálogo, e considerando o estado de uso)
        d. Eff_casco = Vel_med * k / (P_out_BAT * Eff_esc * Eff_motor * Eff_transmissao) -> [FAILED]

B. Estimar a eficiência da entrada Eff_in = Eff_paineis * Eff_MPPT
    1. Eff_paineis
        a. Com base nos dados da rede CAN
        b. Eff_paineis = G * A / P_in_MPPT
    2. Eff_MPPT
        a. Com base nos ensaios do MPPT
        b. Eff_MPPT = P_out_MPPT / P_in_MPPT
