# NOTES:

Tendo:
    - o dado de controle do piloto das provas de 2019 ou 2020,
    - os tempos reais das provas,
    - tendo os dados climáticos durante as competições,
    - tendo conhecimento sobre os componentes internos da embarcação
Então, se o modelo proposto consegue representar a performance da embarcação, será possível parametrizar o modelo utilizando os dados de uma prova, e os dados coletados por meio da
simulação devem condizer com os dados desta, e de outras provas.


# TODO

- [x] Atualizar os equacionamentos e rodar novamente as otimizações dos modelos:
    - [x] Motor: dividimos o motor_k em motor_kv e motor_kq
    - [x] Propulsion: atualizamos as equações
    - [x] ESC: atualizamos as equações
    - [x] PV: atualizamos as equações
- [ ] Re-escrever o equacionamento no arquivo latex baseado no equacionamento de equations_new.ipynb
- [ ] Caso de uso básico: