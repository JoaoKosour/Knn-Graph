from sklearn.neighbors import kneighbors_graph
from collections import defaultdict
from queue import PriorityQueue
from random import randint
from random import choice

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
import math
import heapq

#############################################################################
#                                                                           #
#                              Funçoes Auxiliares                           #
#                                                                           #
#############################################################################
def transform_undirected(adjmatrix):
    for i in range(graph_size):
        for j in range(graph_size):
            if(adjmatrix[i][j] > 0):
                adjmatrix[j][i] = adjmatrix[i][j]
            if(adjmatrix[j][i] > 0):
                adjmatrix[i][j] = adjmatrix[j][i]
    return adjmatrix


class Node:

    def __init__(self, parente=None, posicao=None):
        self.parente = parente
        self.posicao = posicao
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, node_comparacao):
        return self.posicao == node_comparacao.posicao
    
    def __repr__(self):
      return f"{self.posicao} - g: {self.g} h: {self.h} f: {self.f}"

    def __lt__(self, node_comparacao):
      return self.f < node_comparacao.f
    
    def __gt__(self, node_comparacao):
      return self.f > node_comparacao.f

def transform_adjmatrix_to_adjlist(adjlist):
    graph = defaultdict(list)
    edges = set()

    for i, v in enumerate(adjlist):
        for j, u in enumerate(v):
            if u != 0 and frozenset([i, j]) not in edges:
                edges.add(frozenset([i, j]))
                graph[i].append(j)
    return graph


def caminho(node_atual):
    caminho = []
    atual = node_atual
    while atual is not None:
        caminho.append(atual.posicao)
        atual = atual.parente
    return caminho[::-1]


#############################################################################
#                                                                           #
#                                      BFS                                  #
#                                                                           #
#############################################################################
BFS_path = []

def BFS(start, target, adjmatrix, graph_size):
        visited = [False] * graph_size
        q = [start]
 
        visited[start] = True
 
        while q:
            vis = q[0]
 
            BFS_path.append(vis)
            if vis == target:
                return BFS_path
            q.pop(0)
            
            for i in range(graph_size):
                if (adjmatrix[vis][i] == 1 and (not visited[i])):
                    q.append(i)
                    visited[i] = True


#############################################################################
#                                                                           #
#                                      DFS                                  #
#                                                                           #
#############################################################################
DFS_path = []

def DFS(start, target, adjmatrix, graph_size):
        visited = [False] * graph_size
        stack = []

        stack.append(start)
 
        while len(stack):
            start = stack[-1]
            stack.pop()

            if (not visited[start]):
                DFS_path.append(start)
                visited[start] = True
            
            if start == target:
                return DFS_path
        
            
            for i in range(graph_size):
                if (adjmatrix[start][i] == 1 and (not visited[i])):
                    stack.append(i)


#############################################################################
#                                                                           #
#                                  Best First                               #
#                                                                           #
#############################################################################
BSTF_path = []

def best_first_search(source, target, n, graph):
    visited = [0] * n
    visited[source] = True
    pq = PriorityQueue()
    pq.put((0, source))
    while pq.empty() == False:
        u = pq.get()[1]
        BSTF_path.append(u)
        if u == target:
            return BSTF_path
  
        for v, c in graph[u]:
            if visited[v] == False:
                visited[v] = True
                pq.put((c, v))

  
def addedge(i, j, cost, graph):
    graph[i].append((j, cost))


#############################################################################
#                                                                           #
#                                      A Star                               #
#                                                                           #
#############################################################################

def astar(start, end, points, adjmatrix):

    no_inicial = Node(None, start)
    no_inicial.g = no_inicial.h = no_inicial.f = 0

    node_final = Node(None, end)
    node_final.g = node_final.h = node_final.f = 0

    visitar = []
    visitado = []

    heapq.heapify(visitar) 
    heapq.heappush(visitar, no_inicial)

    tratamento_iteracoes_limite_externas = 0
    iteracoes_maximas = (len(adjmatrix[0]) * len(adjmatrix) // 2)

    while len(visitar) > 0:
        tratamento_iteracoes_limite_externas += 1
        # esse if trata caso o programa entre em um loop e rode pra sempre
        if tratamento_iteracoes_limite_externas > iteracoes_maximas:
            return caminho(atual)       
        

        atual = heapq.heappop(visitar)
        visitado.append(atual)

        # caso o node inicial seja o final
        if atual == node_final:
            return caminho(atual)

        children = []
        # Essa é a parte mais importante e diferencial dessa implementação do A*, é a parte que faz o algoritimo funcionar com um grafo KNN
        # nova_posicao percorre todos os pontos mas no primeiro if fazemos uma checagem (usando a matriz de adjascencia do grafo) 
        # que nos diz se o nó esta ligado no nó atual se a verificação falha, o algoritimo "pula" este nó, se não ele é inserido na lista de 
        # nós children
        # a parte mais importante de entender é a verificação no primeiro if, se a posição na matriz de adjascencia é == 0 quer dizer que não temos 
        # ligação
        for nova_posicao in points:
            if adjmatrix[points.index(atual.posicao)][points.index(nova_posicao)] != 1:
                continue
            node_novo = Node(atual, nova_posicao)
            children.append(node_novo)

        for filho in children:
            if len([filho_visitado for filho_visitado in visitado if filho_visitado == filho]) > 0:
                continue

            filho.g = atual.g + 1
            filho.h = ((filho.posicao[0] - node_final.posicao[0]) * 2) + ((filho.posicao[1] - node_final.posicao[1]) * 2)
            filho.f = filho.g + filho.h

            if len([node_aberto for node_aberto in visitar if filho.posicao == node_aberto.posicao and filho.g > node_aberto.g]) > 0:
                continue

            heapq.heappush(visitar, filho)
    return None
    # codigo inspirado no pseudo codigo presente em https://en.wikipedia.org/wiki/A*_search_algorithm
            
#############################################################################
#                                                                           #
#                                   Main                                    #
#                                                                           #
#############################################################################
if __name__ == "__main__":
    graph_size = int(input('Digite o numero de vertices: '))
    number_of_neighbors = int(input('Digite o numero de arestas: '))

    # Gerando pontos aleatórios
    points = []
    for i in range(graph_size):
        points.append((randint(1, graph_size), randint(1, graph_size)))

    # Cria matriz de adjacencias de um grafo direcionado e transforma em não direcionado
    adjmatrix = kneighbors_graph(points, number_of_neighbors).toarray()
    adjmatrix = transform_undirected(adjmatrix)

    # Cria lista de adjacencias
    adjlist = transform_adjmatrix_to_adjlist(adjmatrix)


    #------------------ Plotagem do grafo ------------------#

    # Escrevendo a matriz de adjacencias em um arquivo
    myfile = open('matrix.csv','w')
    count = 0

    print(',', end = '', file = myfile)
    for i in range (graph_size-1):
        print(count, end = ',', file = myfile)
        count = count + 1
    print(count, file = myfile)

    count = 0
    for i in range (graph_size):
        print(count, end = ',', file = myfile)
        count = count + 1
        for j in range (graph_size):
            if j == graph_size-1:
                print(int(adjmatrix[i][j]), file = myfile)
            else:
                print(int(adjmatrix[i][j]), end = ',', file = myfile)

    myfile.close()


    # Criando o grafo
    input_data = pd.read_csv('matrix.csv', index_col=0)
    result_graph = nx.Graph(input_data.values)
    pos = nx.spring_layout(result_graph)
    

    # Escolha aleatoria de 2 nodes para ser o começo e o fim
    startPoint = choice(list(result_graph.edges()))[0]
    endPoint = choice(list(result_graph.edges()))[0]

    print('Inicio: ', startPoint)
    print('Fim: ', endPoint)

    print('1- BFS\n2- DFS\n3- Best First\n4- A*\n0- Sair do programa\n')
    op = 1

    while (op):
        op = int(input('Digite o numero da operacao: '))
        
        # BFS
        if op == 1:
            BFS_start = time.time()
            BFS_result = BFS(startPoint, endPoint, adjmatrix, graph_size)
            BFS_end = time.time()
            
            if BFS_result == None:
                print("Não existe um caminho entre o ponto inicial", startPoint, "e o ponto final", endPoint)

            else:
                print('BFS:')
                print('\t- Caminho: ', BFS_result)
                print('\t- Tempo de execucao: ', (BFS_end - BFS_start))
                print('\t- Tamanho do caminho: ', len(BFS_result))
                print('\n')

                nx.draw(result_graph, pos, with_labels="True", font_size=7, node_size=140)
                nx.draw_networkx_nodes(result_graph, pos, nodelist=BFS_result, node_color='r', node_size=140)
                plt.get_current_fig_manager().set_window_title('BFS')
                plt.show()
        
        # DFS
        elif op == 2:
            DFS_start = time.time()
            DFS_result = DFS(startPoint, endPoint, adjmatrix, graph_size)
            DFS_end = time.time()

            if DFS_result == None:
                print("Não existe um caminho entre o ponto inicial", startPoint, "e o ponto final", endPoint)

            else:
                print('DFS:')
                print('\t- Caminho: ', DFS_result)
                print('\t- Tempo de execucao: ', (DFS_end - DFS_start))
                print('\t- Tamanho do caminho: ', len(DFS_result))
                print('\n')

                nx.draw(result_graph, pos, with_labels="True", font_size=7, node_size=140)
                nx.draw_networkx_nodes(result_graph, pos, nodelist=DFS_result, node_color='r', node_size=140)
                plt.get_current_fig_manager().set_window_title('DFS')
                plt.show()
        
        # Best First
        elif op == 3:
            BestFirst_result = []
            graph = [ [] for j in range(graph_size) ]
            
            for i in range(graph_size):
                for j in range(graph_size):
                    if(adjmatrix[i][j] == 1):
                        addedge(i, j, math.dist(points[j], points[endPoint]), graph)

            BestFirst_start = time.time()
            BestFirst_result = best_first_search(startPoint, endPoint, graph_size, graph)
            BestFirst_end = time.time()

            if DFS_result == None:
                print("Não existe um caminho entre o ponto inicial", startPoint, "e o ponto final", endPoint)

            else:
                print('BSTF:')
                print('\t- Caminho: ', BestFirst_result)
                print('\t- Tempo de execucao: ', (BestFirst_end - BestFirst_start))
                print('\t- Tamanho do caminho: ', len(BestFirst_result))
                print('\n')

                nx.draw(result_graph, pos, with_labels="True", font_size=7, node_size=140)
                nx.draw_networkx_nodes(result_graph, pos, nodelist=BestFirst_result, node_color='r', node_size=140)
                plt.get_current_fig_manager().set_window_title('Best First')
                plt.show()
        
        # A*
        elif op == 4:
            Astar_start = time.time()
            pathAstar = astar(points[startPoint], points[endPoint], points, adjmatrix)
            Astar_end = time.time()

            if pathAstar == None:
                print("Não existe um caminho entre o ponto inicial", startPoint, "e o ponto final", endPoint)
            
            else:
                Astar_result = []
                for coord in pathAstar:
                    for i in range(0, graph_size):
                        if coord == points[i]:
                            Astar_result.append(i)

                print('A*:')
                print('\t- Caminho: ', Astar_result)
                print('\t- Tempo de execucao: ', (Astar_end - Astar_start))
                print('\t- Tamanho do caminho: ', len(Astar_result))
                print('\n')

                nx.draw(result_graph, pos, with_labels="True", font_size=7, node_size=140)
                nx.draw_networkx_nodes(result_graph, pos, nodelist=Astar_result, node_color='r', node_size=140)
                plt.get_current_fig_manager().set_window_title('A*')
                plt.show()

        elif op == 0:
            pass
        
        else:
            print('Operacao invalida!')