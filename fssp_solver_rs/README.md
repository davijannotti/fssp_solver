Solver para o Problema de Flow Shop Scheduling (FSSP)

Este projeto implementa um Algoritmo Memético em Rust para resolver o Problema de Sequenciamento de Tarefas em Flow Shop (FSSP). O objetivo é encontrar a sequência de tarefas que minimiza o tempo total de conclusão, conhecido como makespan (C_max).

A ferramenta é configurável via linha de comando, permitindo o ajuste de parâmetros.
Funcionalidades

    Algoritmo Memético: Combina a exploração global de um algoritmo genético com a intensificação de uma busca local para encontrar soluções de alta qualidade.

    População Híbrida: A população inicial é semeada com soluções gulosas para acelerar a convergência.

    Limites de Execução: O algoritmo pode ser configurado para parar após um número máximo de gerações ou um tempo máximo de execução.

    Geração de Resultados: Salva a melhor solução encontrada (makespan e sequência) em um arquivo de texto.

Execução
    Execute o Solver:
    O programa requer o caminho para um arquivo de instância como argumento principal.

    ./fssp_solver_rs <caminho-para-instancia> [OPÇÕES]

Uso da Linha de Comando

Você pode ver todas as opções disponíveis executando o programa com a flag --help.

./fssp_solver_rs --help

Uso: fssp_solver_rs [OPÇÕES] <INSTANCE_PATH>

Argumentos:
<INSTANCE_PATH>
O caminho para o arquivo da instância FSSP

Opções:
--max-generations <MAX_GENERATIONS>
Número máximo de gerações que o algoritmo irá executar
[padrão: 100]

  --max-duration <MAX_DURATION>
      Duração máxima da execução em segundos. O algoritmo encerrará se exceder este tempo

  --output-dir <OUTPUT_DIR>
      Diretório para salvar o arquivo de resultado
      [padrão: .]

  --population-size <POPULATION_SIZE>
      Tamanho da população
      [padrão: 100]

  --mutation-rate <MUTATION_RATE>
      Taxa de mutação (probabilidade de um indivíduo sofrer mutação)
      [padrão: 0.3]

  --local-search-rate <LOCAL_SEARCH_RATE>
      Taxa de busca local (probabilidade de um indivíduo passar por busca local)
      [padrão: 0.6]

-h, --help
Imprime informação de ajuda

-V, --version
Imprime informação da versão


### Exemplos de Uso

-   **Execução simples com valores padrão:**
    ```sh
    ./fssp_solver_rs ./instances/fssp_instance_05.txt
    ```

-   **Execução com limite de tempo de 60 segundos:**
    ```sh
    ./fssp_solver_rs ./fssp_instance_07.txt --max-duration 60
    ```

-   **Execução com parâmetros personalizados e salvando em um diretório específico:**
    ```sh
    mkdir -p resultados
    ./target/release/fssp_solver_rs ./src/instances/fssp_instance_07.txt \
        --population-size 50 \
        --mutation-rate 0.2 \
        --local-search-rate 0.5 \
        --output-dir ./resultados
    ```

## Parâmetros do Algoritmo e Recomendações

Os valores padrão foram escolhidos como um ponto de partida equilibrado, mas os melhores parâmetros podem variar dependendo da complexidade da instância.

-   `--population-size` **(Padrão: 100)**
    -   **O que faz?**: Define quantos indivíduos (soluções) existem em cada geração.
    -   **Recomendação**: Populações maiores (ex: 50-100) aumentam a diversidade e a capacidade de explorar o espaço de busca, mas tornam cada geração mais lenta. Populações menores convergem mais rápido, mas correm o risco de ficar presas em ótimos locais. O valor **100** é um bom meio-termo.

-   `--max-generations` **(Padrão: 100)**
    -   **O que faz?**: É um critério de parada. O algoritmo para após este número de gerações, a menos que o `--max-duration` seja atingido antes.
    -   **Recomendação**: Para instâncias maiores ou mais complexas, aumente este valor (ex: 500, 1000) para dar ao algoritmo mais tempo para convergir.

-   `--mutation-rate` **(Padrão: 0.3)**
    -   **O que faz?**: Define a probabilidade de um novo indivíduo sofrer uma mutação (troca de duas tarefas). A mutação é crucial para introduzir diversidade e evitar convergência prematura.
    -   **Recomendação**: Uma taxa de **30%** é relativamente alta e incentiva a exploração. Se o seu algoritmo estiver demorando muito para encontrar uma boa solução, você pode tentar diminuir a taxa (ex: 0.1 a 0.2). Se ele converge muito rápido para uma solução que não é ótima, uma taxa mais alta pode ajudar.

-   `--local-search-rate` **(Padrão: 0.6)**
    -   **O que faz?**: Define a probabilidade de um novo indivíduo passar por um processo de busca local (intensificação). Esta é a parte "Memética" do algoritmo, onde as soluções são refinadas ativamente.
    -   **Recomendação**: A busca local é computacionalmente cara, mas muito eficaz. Uma taxa de **60%** garante que uma parte significativa da população seja otimizada a cada geração. Aumentar essa taxa (ex: 0.5) foca mais no refinamento, enquanto diminuí-la favorece a exploração global. O balanço entre a taxa de mutação e a de busca local define o comportamento do algoritmo.

## Arquivos de Saída

Para uma instância chamada `instancia_XX.txt`, o programa gera os seguintes arquivos:

1.  **Arquivo de Resultados**: `resultado_instancia_XX.txt`
    -   Contém o melhor makespan, a melhor sequência encontrada e o tempo de execução.
