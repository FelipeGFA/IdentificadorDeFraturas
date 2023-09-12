# Importar as bibliotecas necessárias
import cv2
import numpy as np
from skimage import filters, morphology, measure

# Importar a biblioteca de aprendizado de máquina
from sklearn.cluster import KMeans

# Ler a imagem da radiografia em escala de cinza
img = cv2.imread("radiografia2.png", cv2.IMREAD_GRAYSCALE)

# Redimensionar a imagem para 600x1000 pixels
img = cv2.resize(img, (600, 1000))

# Aplicar um filtro de suavização para reduzir o ruído
img = cv2.GaussianBlur(img, (5, 5), 0)

# Aplicar um algoritmo de detecção de bordas, como o de Canny
edges = cv2.Canny(img, 50, 150)

# Aplicar um algoritmo de segmentação, como o de Otsu
thresh = filters.threshold_otsu(img)
mask = img > thresh

# Aplicar um algoritmo de morfologia matemática, como o de fechamento
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

# Aplicar um algoritmo de esqueletização, como o de Zhang-Suen
skeleton = morphology.skeletonize(mask)

# Aplicar um algoritmo de detecção de pontos de quebra, como o de Hough
lines = cv2.HoughLinesP(skeleton.astype(np.uint8), 1, np.pi/180, 10, minLineLength=10, maxLineGap=5)
break_points = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    break_points.append((x1, y1))
    break_points.append((x2, y2))

# Adicionar a parte para eliminar os pontos de quebra que estão muito próximos da borda do osso
# Calcular a transformada da distância da máscara do osso
dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)

# Eliminar os pontos de quebra que estão a uma distância menor que um limiar da borda do osso
threshold = 5 # esse valor pode ser ajustado de acordo com a imagem
break_points_filtered = []
for point in break_points:
    x, y = point
    if dist[y, x] > threshold:
        break_points_filtered.append(point)

# Adicionar a parte para limitar o número de retângulos gerados usando o algoritmo de K-means
# Definir o número de clusters (fraturas) que se quer encontrar
# Esse valor pode ser ajustado de acordo com a imagem ou estimado automaticamente
n_clusters = 2

# Aplicar o algoritmo de K-means para agrupar os pontos de quebra filtrados
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(break_points_filtered)
labels = kmeans.labels_

# Desenhar um retângulo ao redor de cada cluster de pontos de quebra
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for i in range(n_clusters):
    # Encontrar os pontos extremos do cluster
    cluster_points = np.array(break_points_filtered)[labels == i]
    min_x = np.min(cluster_points[:, 0])
    min_y = np.min(cluster_points[:, 1])
    max_x = np.max(cluster_points[:, 0])
    max_y = np.max(cluster_points[:, 1])
    # Desenhar um retângulo ao redor do cluster
    cv2.rectangle(img_color, (min_x-10, min_y-10), (max_x+10, max_y+10), (0, 0, 255), 2)

# Redimensionar a imagem colorida para 600x1000 pixels
img_color = cv2.resize(img_color, (600, 1000))

# Mostrar a imagem original e a imagem com os quadrados indicando as fraturas
cv2.imshow("Original", img)
cv2.imshow("Fractures", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()