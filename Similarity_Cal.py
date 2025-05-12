from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np

# 示例数据集，这里用 SMILES 列表表示
dataset1 = ['CCO', 'CC', 'CCOC']
dataset2 = ['CC', 'CCO', 'CCC']

# 步骤 1: 数据加载与处理
mols1 = [Chem.MolFromSmiles(smiles) for smiles in dataset1]
mols2 = [Chem.MolFromSmiles(smiles) for smiles in dataset2]

# 步骤 2: 生成分子指纹
# 这里使用摩根指纹，半径为 2，指纹长度为 1024
fps1 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols1 if mol is not None]
fps2 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols2 if mol is not None]

# 步骤 3: 计算分子间相似性
similarity_matrix = []
for fp1 in fps1:
    row = []
    for fp2 in fps2:
        # 这里使用 Tanimoto 系数计算相似性
        similarity = DataStructs.FingerprintSimilarity(fp1, fp2, metric=DataStructs.TanimotoSimilarity)
        row.append(similarity)
    similarity_matrix.append(row)

# 步骤 4: 汇总相似性结果
# 方法一：平均相似性
average_similarity = np.mean(similarity_matrix)

# 方法二：最大平均相似性（每个分子在另一个数据集中的最大相似性的平均值）
max_similarities_1 = [max(row) for row in similarity_matrix]
max_similarities_2 = [max(col) for col in zip(*similarity_matrix)]
max_avg_similarity = (np.mean(max_similarities_1) + np.mean(max_similarities_2)) / 2

print(f"平均相似性: {average_similarity}")
print(f"最大平均相似性: {max_avg_similarity}")
