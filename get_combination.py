import csv
import os
import random
import re
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from IPython.display import display
from rdkit import Chem
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import AllChem, Draw, MACCSkeys, BondType
from rdkit.Chem.Draw import IPythonConsole, MolDrawing
from rdkit.Chem.Draw.MolDrawing import DrawingOptions

import polygnn_kit.polygnn_kit as pk

import concurrent.futures
import logging
import PURS as pu
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


# In[2]:

def random_sample(smi_list, name_list, label_list, sample_size):
	"""
	随机采样函数，支持上采样。

	参数:
		smi_list (list): 第一个列表
		name_list (list): 第二个列表
		label_list (list): 第三个列表
		sample_size (int): 指定的采样数量

	返回:
		tuple: 包含采样后的 smi_list, name_list, label_list 的元组
	"""
	# 如果原列表为空，直接返回空列表
	if not smi_list or not name_list or not label_list:
		return [], [], []

	# 确保三个列表长度一致
	assert len(smi_list) == len(name_list) == len(label_list), "三个列表长度必须一致"

	# 生成索引列表
	indices = list(range(len(smi_list)))

	# 如果原列表的长度大于或等于指定的采样数量，使用 random.sample
	if len(smi_list) >= sample_size:
		sampled_indices = random.sample(indices, sample_size)
	# 如果原列表的长度小于指定的采样数量，使用 random.choices 上采样
	else:
		sampled_indices = random.choices(indices, k=sample_size)

	# 根据采样的索引提取对应的元素
	sampled_smi = [smi_list[i] for i in sampled_indices]
	sampled_name = [name_list[i] for i in sampled_indices]
	sampled_label = [label_list[i] for i in sampled_indices]

	return sampled_smi, sampled_name, sampled_label
	
def Classification_by_ring(smi):
	num_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '%']
	num = []
	for j in smi:
		if j in num_list:
			num.append(j)
	if len(num) > 4:
		ring_type = 'fused'
	elif len(num) == 4:
		ring_type = 'double'
	elif len(num) == 2:
		ring_type = 'single'
	elif len(num) == 0:
		ring_type = 'bratch'
	else:
		ring_type = 'unknown'
	return ring_type


# In[3]:


def check_elements(smiles):
	# 创建分子对象
	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		return '无效的 SMILES 字符串'
	
	# 获取分子中原子的元素符号
	elements = set([atom.GetSymbol() for atom in mol.GetAtoms()])
	return elements


# In[4]:


def common_elements(df, top_n=5):
	# 对 DataFrame 中的每个 SMILES 应用 check_elements 函数
	df['Elements'] = df['smiles'].apply(check_elements)
	
	# 将所有元素集合展开为一个列表
	all_elements = [element for elements_set in df['Elements'] for element in elements_set if element !='C']
	
	# 统计元素的频率
	element_counts = Counter(all_elements)
	
	# 找出出现频率最高的 top_n 种元素
	most_common_elements = element_counts.most_common(top_n)
	common_elements_list = [element for element, count in most_common_elements]
	
	# 更新 DataFrame 中的 'Elements' 列，仅保留 common_elements_list 中的元素
	df['Elements'] = df['Elements'].apply(lambda x: sorted([element for element in x if element in common_elements_list]))
	
	return common_elements_list, df


# In[5]:


def add_polymer_type(df):
	# 将 'Elements' 列的集合转换为排序后的字符串
	df['Elements_str'] = df['Elements'].apply(lambda x: '-'.join(sorted(x)))
	
	# 组合 'ring_type' 和 'Elements_str' 列，形成 'polymer_type' 列
	df['polymer_type'] = df['ring_type'] + '-' + df['Elements_str']
	
	# 删除中间步骤创建的 'Elements_str' 列
	df.drop(columns=['Elements_str'], inplace=True)
	
	return df


# In[6]:


def split_smiles_list(smiles_list):
	"""
	将 single_list 中的每个字符串按 '+' 分隔成子字符串列表，并将这些子字符串列表存储在一个新列表中

	:param smiles_list: 包含 SMILES 字符串的列表
	:return: 包含分隔后子字符串列表的列表
	"""
	split_list = []
	for smiles in smiles_list:
		# 去掉开头的 '+' 号
		if smiles.startswith('+'):
			smiles = smiles[1:]
		# 按 '+' 分隔
		split_smiles = smiles.split('+')
		split_list.append(split_smiles)
	return split_list


# In[7]:


# 定义一个函数来处理每一列，生成新的列
def process_column(df, column_name):
	# 统计每列中不同元素的数量
	unique_elements = df[column_name].unique()
	unique_elements.sort()
	# 创建一个字典，将每个元素映射到一个唯一的编号
	element_to_num = {element: i+1 for i, element in enumerate(unique_elements)}
	# 添加新列
	new_column_name = f"{column_name}-num"
	df[new_column_name] = df[column_name].map(element_to_num)
	# 求最大索引
	max_index = max(element_to_num.values())
	# 归一化新列
	df[new_column_name] = df[new_column_name] / max_index
	return unique_elements


# In[8]:


def set_mark(smi):
	mol = Chem.MolFromSmiles(smi)
	index_list = []
	for atom in mol.GetAtoms():
		if atom.GetDegree() == 1:
			if atom.GetSymbol() == "C" or atom.GetSymbol() == "N":
				# 检查是否与任何其他原子形成三键
				if all(bond.GetBondType() != Chem.BondType.TRIPLE for bond in atom.GetBonds()):
					atom_idx = atom.GetIdx()
					index_list.append(atom_idx)
	mark_list = []
	if len(index_list) >= 2:
		all_combinations = list(itertools.combinations(index_list, 2))
		for idx in all_combinations:
			mol2 = Chem.MolFromSmiles(smi)
			for i in idx:
				atom = mol2.GetAtomWithIdx(i)
				atom.SetAtomMapNum(1)
			marked_smiles = Chem.MolToSmiles(mol2)
			mark_list.append(marked_smiles)
	elif len(index_list) == 1:
		mol2 = Chem.MolFromSmiles(smi)
		atom = mol2.GetAtomWithIdx(index_list[0])
		atom.SetAtomMapNum(1)
		marked_smiles = Chem.MolToSmiles(mol2)
		mark_list.append(marked_smiles)
	return mark_list


# In[9]:


def select_smiles_for_combination(df, combo_types, repeat_num=3):
	"""
	根据组合类型，随机选择SMILES
	:param df: 包含SMILES和polymer_type的DataFrame
	:param combo_types: 需要选择的polymer_type列表
	:param repeat_num: 每种polymer_type最多选择的SMILES数量
	:return: 包含所选SMILES的列表
	"""
	selected_smiles = []
	
	for type_ in combo_types:
		# 筛选出 polymer_type 等于当前类型的行
		filtered_df = df[df['polymer_type'] == type_]
		
		if filtered_df.empty:
			return f"No matching polymer_type found for: {type_}"
		
		# 随机选择1到repeat_num个结构
		num_to_select = random.randint(1, repeat_num)
		random_rows = filtered_df.sample(n=num_to_select,replace=True)
		
		# 获取SMILES并添加到结果列表中
		selected_smiles.extend(random_rows['SMILES'].values)
	
	return selected_smiles

from itertools import product


def select_by_ratio(lst,ratio):
	if ratio > 1:
		print("Error:ratio must smaller than 1!!!")
	num_to_select = max(1, round(len(lst) * ratio))
	# 随机选择指定数量的元素
	selected_elements = random.sample(lst, num_to_select)
	return selected_elements

def combine_lists(A, B, C,ratio=1):
	"""
	穷举出 bratch, single, double 中元素的所有组合，并将每个组合的子列表合并成一个单一的子列表

	:param A: 列表 A
	:param B: 列表 B
	:param C: 列表 C
	:return: 包含所有合并后子列表的列表
	"""
	logging.info("按比率筛选组合元素...")
	if ratio < 1:
		A = select_by_ratio(A,ratio)
		B = select_by_ratio(B,ratio)
		C = select_by_ratio(C,ratio)
	logging.info("随机组合列表...")
	combined_list = []
	for a, b, c in product(A, B, C):
		# 合并子列表
		combined_sublist = a + b + c
		filtered_sublist = [item for item in combined_sublist if 'no' not in item]
		combined_list.append(filtered_sublist)
	return combined_list

def generate_all_possible_connections(fragments):
	# 对于选取的片段，生成所有可能的排列
	permutations = list(itertools.permutations(fragments))
	all_results = []
	for perm in permutations:
		# 对每个排列中的每个片段调用 set_mark 函数，获取所有可能的标记方式
		marked_fragments = [set_mark(fragment) for fragment in perm]
		# 生成该排列下的所有可能的标记组合
		combinations = list(itertools.product(*marked_fragments))
		all_results.extend(combinations)
	
	return all_results

def find_and_offset_marked_atom_indices(fragments):
	marked_atoms = []
	total_atoms = 0	 # Tracking the total atoms processed so far
	for fragment in fragments:
		mol = Chem.MolFromSmiles(fragment)
		marked_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 1]
		# Offset indices by the total number of atoms from previous fragments
		for i in range(len(marked_indices)):
			marked_indices[i] += total_atoms
		total_atoms += mol.GetNumAtoms()  # Update the total number of atoms processed
		marked_atoms.append(marked_indices)
		num = num + 1
	return marked_atoms

def combin_n_mol(smi_list):
	mol_list = [Chem.MolFromSmiles(i) for i in smi_list]
	com_mol = mol_list[0]
	for i in mol_list[1:]:
		com_mol = Chem.CombineMols(com_mol, i)
	return com_mol

def find_and_replace(text, pattern1=r'\[CH(\d+):1\]', pattern2='C', replace=0):
	"""
	使用正则表达式找到并替换所有符合模式的字符串

	:param text: 原始字符串
	:param pattern1: 匹配模式，例如 '[CHn:1]'，其中 n 是数字
	:param pattern2: 替换模式，例如 'C'
	:param replace: 替换次数，0 表示替换所有，1 表示只替换第一个匹配
	:return: 替换后的字符串
	"""
	def replacer(match):
		n = match.group(1)
		return pattern2

	if replace == 0:
		text = re.sub(pattern1, replacer, text)
	elif replace == 1:
		text = re.sub(pattern1, replacer, text, count=1)
	
	return text

def get_id_bymark(combo,mark):
	"""
	找到标记原子的原子索引编号,然后将原子的标记设定为零
	"""
	for at in combo.GetAtoms():
		if at.GetAtomMapNum()==mark:
			return at.GetIdx()
		
def combinefragbydifferentmatrix(fragsmi_list,matrix):
	com_mol = combin_n_mol(fragsmi_list)
	edcombo = add_bond(com_mol,matrix)
	back = edcombo.GetMol()
	smi= Chem.MolToSmiles(back)
	mol = Chem.MolFromSmiles(smi)
	if not mol:
		smi = find_and_replace(smi, pattern1=r'\[CH(\d+):(\d+)\]', pattern2='C', replace=0)
		mol = Chem.MolFromSmiles(smi)
		if not mol:
			pass
		else:
			return smi
	else:
		return smi
	
def add_bond(combo,adj_matrix):
	
	for i in adj_matrix:
		edcombo = Chem.EditableMol(combo)
		amark = get_id_bymark(combo,i[0])
		a_atom = combo.GetAtomWithIdx(amark)
		a_nei = [nei.GetIdx() for nei in a_atom.GetNeighbors()][0]
		bmark = get_id_bymark(combo,i[1])
		b_atom = combo.GetAtomWithIdx(bmark)
		b_nei = [nei.GetIdx() for nei in b_atom.GetNeighbors()][0]
		edcombo.AddBond(a_nei,b_nei,order=Chem.rdchem.BondType.SINGLE)
		combo = edcombo.GetMol()
		
	for i in adj_matrix:
		amark = get_id_bymark(combo,i[0])
		edcombo = Chem.EditableMol(combo)
		edcombo.RemoveAtom(amark)
		back = edcombo.GetMol()
		bmark=get_id_bymark(back,i[1])
		edcombo=Chem.EditableMol(back)
		edcombo.RemoveAtom(bmark)
		combo = edcombo.GetMol()
		
	return edcombo

def get_neiid_bysymbol(combo,symbol):
	for at in combo.GetAtoms():
		if at.GetSymbol()==symbol:
			at_nei=at.GetNeighbors()
			return at_nei.GetIdx()
		
def get_nei_idx_by_idx(combo, idx):
	"""
	根据原子索引找到该原子的邻居索引

	:param combo: RDKit 分子对象
	:param idx: 原子索引
	:return: 原子的邻居索引列表
	"""
	atom = combo.GetAtomWithIdx(idx)
	nei_indices = [nei.GetIdx() for nei in atom.GetNeighbors()]
	return nei_indices[0]

def rename_mark(smiles_list):
	mark = 1
	new_list = []
	lists = []
	for smi in smiles_list:
		l = []
		mol = Chem.MolFromSmiles(smi)
		for atom in mol.GetAtoms():
			if atom.GetAtomMapNum() == 1:
				atom.SetAtomMapNum(mark)
				l.append(mark)
				mark = mark + 1
		smi = Chem.MolToSmiles(mol)
		new_list.append(smi)
		lists.append(l)
	return new_list,lists

def generate_combinations(lists):
	#对于每种片段排列，生成所有的连接方式
	if len(lists) < 2:
		return []
	# 从第一个列表中随机选择一个元素
	first_element = random.choice(lists[0])
	# 从第二个列表中随机选择一个元素
	second_element = random.choice(lists[1])
	remaining_second = [x for x in lists[1] if x != second_element]
	# 生成第一个二元列表
	pairs = [[first_element, second_element]]
	# 递归处理剩余的列表
	if len(remaining_second) > 0:
		pairs.extend(generate_combinations([remaining_second] + lists[2:]))

	return pairs

def remove_atom_labels_from_smiles(smiles):
	# 解析SMILES字符串，生成一个分子对象
	mol = Chem.MolFromSmiles(smiles)
	
	
	# 遍历分子中的所有原子，去掉原子标记
	for atom in mol.GetAtoms():
		atom.SetAtomMapNum(0)  # 设置原子标记为0，相当于去掉标记
 
	new_smiles = Chem.MolToSmiles(mol)
	return new_smiles

def remove_all_parentheses_C_from_smiles(smiles):
	# 使用正则表达式一次性替换所有 "(C)"
	cleaned_smiles = re.sub(r'\(C\)', '', smiles)
	return cleaned_smiles
def replace_patterns(smi):
	if '.' in smi:
		return None
	#去除原子标号
	smi = remove_atom_labels_from_smiles(smi)
	# 定义需要查找和替换的模式列表
	patterns = [
		(r'\(Cc(\d+)', r'(C=Cc\1'),
		(r'\(CC(\d+)', r'(C=CC\1'),
		(r'\(CN(\d+)', r'(C=CN\1'),
		(r'\(CCN(\d+)', r'(C=CN\1'),
		(r'\(CCc(\d+)', r'(C=Cc\1'),
		(r'\(CCC(\d+)', r'(C=CC\1'),
		(r'\(CC(C)C(\d+)', r'(CC\1'),
		(r'\(C=C\)', '(C=CC)'),
		#(r'\(C\)', '')
	]
	
	# 遍历每个模式，先检查是否存在，如果存在则进行替换
	for pattern, replacement in patterns:
		if re.search(pattern, smi):
			smi = re.sub(pattern, replacement, smi)
			
	if smi[:3] == 'C=C':
		smi = 'C'+smi
		
	return smi

def molecular_to_polymer(smi):
	try:
		mol = Chem.MolFromSmiles(smi)
	except:
		return None
	pm = None
	if mol:
		mark_list = set_mark(smi)
		if len(mark_list)>0:
			for i in mark_list:
				
				#fig = Draw.MolToImage(i[1], size=(1000,1000), kekulize=True)
				#display(fig)
				pattern = r"\[[^]]*:[^]]*\]" 
				output_str = re.sub(pattern, "[*]", i)
				if "[g]" in output_str:
					polymer_class = pk.LadderPolymer
				else:
					polymer_class = pk.LinearPol
					try:
						lp = polymer_class(output_str)
						pm = lp.multiply(1).PeriodicMol()
					except:
						pass
						
				if pm:
					return output_str
				


def process_combination(ring_df,combination,bond_type="single"):
	"""
	bond_type可选择“single”和“single_double”
	其中“single”将Polymer-unit以单键的方式直连；“single_double”则将Polymer_unit以单双键交替的方式连接
	"""
	new_smi_list = []
	fragments = select_smiles_for_combination(ring_df, combination)
	all_connections = generate_all_possible_connections(fragments)
	for connection in all_connections:
		connection = list(connection)
		connection, lists = rename_mark(connection)
		pairs = generate_combinations(lists)
		if len(pairs) == len(connection) - 1:
			str_smi = combinefragbydifferentmatrix(connection, pairs)
			if str_smi:
				mol = Chem.MolFromSmiles(str_smi)
				if bond_type == "single_double":
					str_smi = replace_patterns(str_smi)
					if str_smi:
						smi = replace_patterns(str_smi)
						if smi:
							if smi:
								mol = Chem.MolFromSmiles(smi)
				if mol:
					smi = molecular_to_polymer(str_smi)
					if smi:
						new_smi_list.append(smi)
						del str_smi,smi,mol,connection,pairs
	return new_smi_list

def save_to_csv(data, file_name):
	df = pd.DataFrame(data, columns=['smiles'])
	df.to_csv(file_name, index=False)
	


# In[10]:


def main(combined_list,ring_df):
	new_str_list = []
	for combination in combined_list:
		smi_list = process_combination(ring_df,combination)
		new_str_list += smi_list
	new_str = list(set(new_str_list))
	return new_str


# In[11]:


def parallel_main(combined_list,ring_df,num_workers=1,output="new_data/output"):
	"""
	当生成的组合数较多时使用，利用并行加速数据生成
	"""
	total_new_list = []
	count = 0
	save_interval = 1000
	base_output_file = output
	file_counter = 0
	
	with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
		futures = []
		for combination in combined_list:
			futures.append(executor.submit(process_combination, ring_df,combination))
		
		for future in concurrent.futures.as_completed(futures):
			results = future.result()
			if results:
				total_new_list += results
				count += len(results)
			
			# 每处理5000个结果，保存到CSV文件并重置列表
			if count >= save_interval:
				file_name = f"{base_output_file}_{file_counter}.csv"
				save_to_csv(total_new_list, file_name)
				logging.info(f"Saved {count} results to {file_name}")
				total_new_list = []
				count = 0
				file_counter += 1
	# 保存剩余的结果
	if total_new_list:
		file_name = f"{base_output_file}_{file_counter}.csv"
		save_to_csv(total_new_list, file_name)
		logging.info(f"Saved remaining {len(total_new_list)} results to {file_name}")
	print("Processing complete.")


# In[18]:



def get_ring_df(input_file,smiles_index=0,label_index=None,threshold=None,filter_direction='above',sample_size=None):

	logging.info("Step 1: 开始处理输入数据...")
	smi_list,name_list,label_list = pu.processing_data(input_file,smiles_index,label_index,sample_size)
	logging.info("Step 2: 获取polymer-unit...")
	ring_total_list2,total_neighbor_data = pu.get_pu(smi_list,name_list)
	logging.info("Step 3: 分析polymer-unit类型...")
	ring_df = pd.DataFrame(ring_total_list2, columns=['smiles'])
	ring_df['ring_type'] = ring_df['smiles'].apply(Classification_by_ring)
	
	if threshold:
		if filter_direction=='above':
			name_list = [name for name, label in zip(name_list, label_list) if label > threshold]
		if filter_direction=='below':
			name_list = [name for name, label in zip(name_list, label_list) if label <= threshold]
	index_frame=pu.get_index_list(name_list,ring_total_list2,total_neighbor_data)
	most_common_elements,ring_df = common_elements(ring_df)
	ring_df = add_polymer_type(ring_df)
	
	return ring_df,index_frame
	
def get_combination(ring_df,index_frame,combin_ratio=1):
	
	bratch_list=[]
	bratch_list2=[]
	single_list=[]
	single_list2=[]
	double_fused_list=[]
	double_fused_list2=[]
	a=index_frame.shape[1]
	b=index_frame.shape[0]
	n=0
	while n < b:
		j=0
		bratch=[]
		single=[]
		double_fused=[]
		all_carbon_bratch=True
		while j < a:
			#num=index_frame[j][n]
			num=index_frame.iloc[n, j]
			if num!='none':
				num=int(num)
				type_name=ring_df.loc[num]['ring_type']
				str_name=ring_df.loc[num]['polymer_type']
				if type_name =='bratch':
					bratch.append(str_name)
				elif type_name=='single':
					single.append(str_name)
				elif type_name=='double':
					double_fused.append(str_name)
				elif type_name=='fused':
					double_fused.append(str_name)
			j=j+1
		bratch.sort()
		single.sort()
		double_fused.sort()
		
		br_set=[]
		for br in bratch:
			if br not in br_set:
				br_set.append(br)
		bratch_list2.append(br_set)
				
		single_set=[]
		for s in single:
			if s not in single_set:
				single_set.append(s)
		single_list2.append(single_set)
				
		double_fused_set=[]
		for d in double_fused:
			if d not in double_fused_set:
				double_fused_set.append(d)
		double_fused_list2.append(double_fused_set)
		
		br_name=''
		single_name=''
		double_fused_name=''
		for br in br_set:
			br_name=br_name+'+'+br
		for s in single_set:
			single_name=single_name+'+'+s
		for d in double_fused_set:
			double_fused_name=double_fused_name+'+'+d
		if len(br_name)==0:
			br_name='no_bracth'
		if len(single_name)==0:
			single_name='no_single'
		if len(double_fused_name)==0:
			double_fused_name='no_fused'
		bratch_list.append(br_name)
		single_list.append(single_name)
		double_fused_list.append(double_fused_name)
		n=n+1
	
	unit_type_df=pd.DataFrame(index=index_frame.index)
	unit_type_df['bracth']=bratch_list
	unit_type_df['single']=single_list
	unit_type_df['double/fused']=double_fused_list
	
	# 处理每一列
	logging.info("Step 4:生成所有可能的组合类型...")
	logging.info("处理支链部分...")
	bratch_list = split_smiles_list(process_column(unit_type_df, 'bracth'))
	logging.info("处理单环部分...")
	single_list = split_smiles_list(process_column(unit_type_df, 'single'))
	logging.info("处理稠环部分...")
	double_list = split_smiles_list(process_column(unit_type_df, 'double/fused'))
	logging.info("生成组合类型...")
	
	combined_list = combine_lists(bratch_list, single_list, double_list)
	
	return combined_list



