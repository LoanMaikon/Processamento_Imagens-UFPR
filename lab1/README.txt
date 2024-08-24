Nome: Luan Marko Kujavski
GRR: GRR20221236

Na primeira execução em um dataset, guardamos os histogramas em um JSON para que não tenhamos de executar para termos essas informações novamente. É importante ressaltar que os histogramas são normalizados antes de serem guardados.
Um ponto importante é que são gerados e usados 3 histogramas (RGB) para cada imagem. O protocolo de testes para comparação entre duas imagens é comparar cada canal de uma imagem com o canal correspondente da outra imagem e calcular o melhor dos três valores que compareHist retorna (levando em conta que existem métodos que um valor menor é melhor e vice-versa)

Foi adicionado um novo parâmetro para a execução do programa na linha de comando. Você pode colocar ou não o K que deseja.
Usage: python3 histograma.py <method> <dataset_folder> <K>
Caso nenhum K seja passado, o programa usará o valor 1 seguindo um teste de validação com 40% dos dados disponíveis.

Esse teste de validação foi feito em Kvalidation.py. Se validation_size não for passado será usado 40% do dataset como validação.
Usage: python3 Kvalidation.py <dataset_folder> <validation_size>
O protocolo seguido foi testar K = [1, 2, 3, 4] no dataset passado. A partir do 5 o K começa a não fazer sentido, pois há somente outras 4 imagens que correspondem à imagem de teste. O resultado de cada K é a média das acurácias de cada um dos métodos

Usando a seed 69, o resultado foi:
For K = 1, the Mean Accuracy is 0.56
For K = 2, the Mean Accuracy is 0.56
For K = 3, the Mean Accuracy is 0.48
For K = 4, the Mean Accuracy is 0.46
*Como o dataset é extremamente pequeno, esses testes de validação sofrem grande variação

Outro ponto importante é que minha implementação tem uma função que decide a qual classe o objeto pertence em caso de empate. Nesses casos, a classe em que a distância calculada é menor é escolhida. Sendo assim, para K = 1 e K = 2, os resultados são os mesmos, já que sempre será escolhida a classe mais próxima.

Também, para diminuir o ruído de cores irrelevantes para a classificação de uma imagem, removi manualmente as intensidades 0 e 255 de todos os histogramas gerados
