#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from scipy.linalg import sqrtm
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances
import fcd
from tqdm import tqdm  # 用于进度条
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])

model= fcd.load_ref_model()

# 可修改的参数
INPUT_CSV = "new_data/opv.csv"  # 生成的 SMILES 数据文件
REFERENCE_CSV = "input/opv_processed.csv"  # 参考数据文件（若不需要，可设为 None）
SAVE_CSV = True  # 是否保存结果到 CSV 文件
OUTPUT_VALID = "output/opv_purs_valid_smiles.csv"  # 有效 SMILES 输出文件
OUTPUT_UNIQUE = "output/opv_purs_unique_smiles.csv"  # 唯一 SMILES 输出文件
OUTPUT_NOVEL = "output/opv_purs_novel_smiles.csv"  # 新颖 SMILES 输出文件

def descriptors(smiles):
    """从 SMILES 字符串中提取一组物理化学描述符"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

   # 提取描述符
    desc = [
        Descriptors.MolWt(mol),  # 分子量
        Descriptors.TPSA(mol),  # 拓扑极性表面积
        Descriptors.MolLogP(mol),  # LogP
        Descriptors.NumHDonors(mol),  # 氢键供体数量
        Descriptors.NumHAcceptors(mol),  # 氢键受体数量
        Lipinski.NumRotatableBonds(mol),  # 可旋转键数量
        Lipinski.NumAromaticRings(mol),  # 芳香环数量
    ]
    return np.array(desc)

def check_validity(smiles_list):
    valid_smiles = [s for s in smiles_list if Chem.MolFromSmiles(s) is not None]
    validity = len(valid_smiles) / len(smiles_list) 
    return validity, valid_smiles


def check_uniqueness(valid_smiles):
    unique_smiles = list(set(valid_smiles))
    uniqueness = len(unique_smiles) / len(valid_smiles)
    return uniqueness, unique_smiles


def check_novelty(unique_smiles, reference_smiles):
    novel_smiles = [s for s in unique_smiles if s not in reference_smiles]
    novelty = len(novel_smiles) / len(unique_smiles)
    return novelty, novel_smiles


def calculate_intdivp(smiles_list, p=2, fingerprint_type="Morgan"):
    """计算内部多样性（IntDivp）"""
    if len(smiles_list) < 2:
        return 0.0

    # 生成分子指纹
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            if fingerprint_type == "Morgan":
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            else:
                fp = AllChem.RDKFingerprint(mol)
            fingerprints.append(fp)

    # 计算谷本相似度
    similarities = []
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarities.append(sim)

    # 计算 IntDivp
    if not similarities:
        return 0.0
    return 1 - np.power(np.mean(np.power(similarities, p)), 1 / p)


def calculate_fcd(smiles_list, reference_smiles):
    """计算 Frechet ChemNet Distance (FCD)"""
    return fcd.get_fcd(smiles_list, reference_smiles,model=model)

def calculate_kl_divergence(smiles_list, reference_smiles):
    """计算 KL 散度"""
    # 计算描述符的分布
    gen_dist = np.array([descriptors(s) for s in smiles_list if Chem.MolFromSmiles(s)])
    ref_dist = np.array([descriptors(s) for s in reference_smiles if Chem.MolFromSmiles(s)])

    # 计算 KL 散度
    return entropy(gen_dist.mean(axis=0), ref_dist.mean(axis=0))


def main():
    df = pd.read_csv(INPUT_CSV)
    if 'smiles' not in df.columns:
        raise ValueError("Input CSV must contain a 'smiles' column.")

    smiles_list = df['smiles'].dropna().tolist()
    validity, valid_smiles = check_validity(smiles_list)
    uniqueness, unique_smiles = check_uniqueness(valid_smiles)

    if REFERENCE_CSV:
        ref_df = pd.read_csv(REFERENCE_CSV)
        if 'smiles' not in ref_df.columns:
            raise ValueError("Reference CSV must contain a 'smiles' column.")
        reference_smiles = ref_df['smiles'].dropna().tolist()
        novelty, novel_smiles = check_novelty(unique_smiles, reference_smiles)
    else:
        novelty, novel_smiles = None, None
    
    
    logging.info(f"validity: {validity:.4f}")
    logging.info(f"uniqueness: {uniqueness:.4f}")
    logging.info(f"novelty: {novelty:.4f}")
    # 计算 IntDivp
    intdivp = calculate_intdivp(unique_smiles)
    logging.info(f"Internal Diversity (IntDivp): {intdivp:.4f}")

    # 计算 FCD
    if REFERENCE_CSV:
        fcd = calculate_fcd(unique_smiles, reference_smiles)
        logging.info(f"Frechet ChemNet Distance (FCD): {fcd:.4f}")

    # 计算 KL 散度
    if REFERENCE_CSV:
        kl_divergence = calculate_kl_divergence(unique_smiles, reference_smiles)
        logging.info(f"KL Divergence: {kl_divergence:.4f}")

    if SAVE_CSV:
        pd.DataFrame({'valid_smiles': valid_smiles}).to_csv(OUTPUT_VALID, index=False)
        pd.DataFrame({'unique_smiles': unique_smiles}).to_csv(OUTPUT_UNIQUE, index=False)
        if novelty is not None:
            pd.DataFrame({'novel_smiles': novel_smiles}).to_csv(OUTPUT_NOVEL, index=False)
            
def descriptors(smiles):
    """从 SMILES 字符串中提取一组物理化学描述符"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

   # 提取描述符
    desc = [
        Descriptors.MolWt(mol),  # 分子量
        Descriptors.TPSA(mol),  # 拓扑极性表面积
        Descriptors.MolLogP(mol),  # LogP
        Descriptors.NumHDonors(mol),  # 氢键供体数量
        Descriptors.NumHAcceptors(mol),  # 氢键受体数量
        Lipinski.NumRotatableBonds(mol),  # 可旋转键数量
        Lipinski.NumAromaticRings(mol),  # 芳香环数量
    ]
    return np.array(desc)

def check_validity(smiles_list):
    valid_smiles = [s for s in smiles_list if Chem.MolFromSmiles(s) is not None]
    validity = len(valid_smiles) / len(smiles_list) 
    return validity, valid_smiles


def check_uniqueness(valid_smiles):
    unique_smiles = list(set(valid_smiles))
    uniqueness = len(unique_smiles) / len(valid_smiles)
    return uniqueness, unique_smiles


def check_novelty(unique_smiles, reference_smiles):
    novel_smiles = [s for s in unique_smiles if s not in reference_smiles]
    novelty = len(novel_smiles) / len(unique_smiles)
    return novelty, novel_smiles


def calculate_intdivp(smiles_list, p=2, fingerprint_type="Morgan"):
    """计算内部多样性（IntDivp）"""
    if len(smiles_list) < 2:
        return 0.0

    # 生成分子指纹
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            if fingerprint_type == "Morgan":
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            else:
                fp = AllChem.RDKFingerprint(mol)
            fingerprints.append(fp)

    # 计算谷本相似度
    similarities = []
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarities.append(sim)

    # 计算 IntDivp
    if not similarities:
        return 0.0
    return 1 - np.power(np.mean(np.power(similarities, p)), 1 / p)


def calculate_fcd(smiles_list, reference_smiles):
    """计算 Frechet ChemNet Distance (FCD)"""
    return fcd.get_fcd(smiles_list, reference_smiles,model=model)

def calculate_kl_divergence(smiles_list, reference_smiles):
    """计算 KL 散度"""
    # 计算描述符的分布
    gen_dist = np.array([descriptors(s) for s in smiles_list if Chem.MolFromSmiles(s)])
    ref_dist = np.array([descriptors(s) for s in reference_smiles if Chem.MolFromSmiles(s)])

    # 计算 KL 散度
    return entropy(gen_dist.mean(axis=0), ref_dist.mean(axis=0))


def main():
    df = pd.read_csv(INPUT_CSV)
    if 'smiles' not in df.columns:
        raise ValueError("Input CSV must contain a 'smiles' column.")

    smiles_list = df['smiles'].dropna().tolist()
    validity, valid_smiles = check_validity(smiles_list)
    uniqueness, unique_smiles = check_uniqueness(valid_smiles)

    if REFERENCE_CSV:
        ref_df = pd.read_csv(REFERENCE_CSV)
        if 'smiles' not in ref_df.columns:
            raise ValueError("Reference CSV must contain a 'smiles' column.")
        reference_smiles = ref_df['smiles'].dropna().tolist()
        novelty, novel_smiles = check_novelty(unique_smiles, reference_smiles)
    else:
        novelty, novel_smiles = None, None
    
    
    logging.info(f"validity: {validity:.4f}")
    logging.info(f"uniqueness: {uniqueness:.4f}")
    logging.info(f"novelty: {novelty:.4f}")
    # 计算 IntDivp
    intdivp = calculate_intdivp(unique_smiles)
    logging.info(f"Internal Diversity (IntDivp): {intdivp:.4f}")

    # 计算 FCD
    if REFERENCE_CSV:
        fcd = calculate_fcd(unique_smiles, reference_smiles)
        logging.info(f"Frechet ChemNet Distance (FCD): {fcd:.4f}")

    # 计算 KL 散度
    if REFERENCE_CSV:
        kl_divergence = calculate_kl_divergence(unique_smiles, reference_smiles)
        logging.info(f"KL Divergence: {kl_divergence:.4f}")

    if SAVE_CSV:
        pd.DataFrame({'valid_smiles': valid_smiles}).to_csv(OUTPUT_VALID, index=False)
        pd.DataFrame({'unique_smiles': unique_smiles}).to_csv(OUTPUT_UNIQUE, index=False)
        if novelty is not None:
            pd.DataFrame({'novel_smiles': novel_smiles}).to_csv(OUTPUT_NOVEL, index=False)
            
if __name__ == "__main__":
    main()

