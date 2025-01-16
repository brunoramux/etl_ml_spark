# Queries do SQLite

SELECT prediction AS cluster, COUNT(*) AS contagem
FROM tb_clusters
GROUP BY prediction

CREATE INDEX "idx_pred" ON "tb_clusters" ("prediction");

SELECT prediction AS cluster, COUNT(*) AS contagem, ROUND(AVG(valor),2) AS media
FROM tb_clusters
GROUP BY prediction

SELECT prediction AS cluster, COUNT(*) AS contagem, ROUND(AVG(valor),2) AS media
FROM tb_clusters
WHERE anomalia = 1
GROUP BY prediction
ORDER BY media DESC

SELECT ano, mes, AVG(valor) AS media
FROM tb_clusters
WHERE anomalia = 1 AND prediction = 1
GROUP BY ano, mes
ORDER BY media DESC

SELECT ano, mes, AVG(valor) AS media
FROM tb_clusters
WHERE anomalia = 1 AND prediction = 0
GROUP BY ano, mes
ORDER BY media DESC

SELECT ano, mes, AVG(valor) AS media
FROM tb_clusters
WHERE anomalia = 1 AND prediction = 2
GROUP BY ano, mes
ORDER BY media DESC