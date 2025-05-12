# %load get_new_str
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from IPython.display import display
from rdkit.Chem import Draw
import random
import itertools
import re
from rdkit.Chem import BondType
import polygnn_kit.polygnn_kit as pk
import concurrent.futures
import logging
from itertools import product
import random
import get_combination as get_com
import evaluate_data as ev
import pickle as pkl
import concurrent.futures
import gc 

random.seed(42)
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


input_file = 'input/log_p_last10.csv'
output_file = 'new_data/log_p_last10.csv'
data_num=500
type_num=200
max_smi=100
sample_size = 10
threshold=None
label_index=None
smiles_index = 'smiles'
filter_direction='below'#'below'or'above'
num_workers = 16
replace = False


def random_sample(combined_list, sample_size):
	"""
	随机采样函数，支持上采样。

	参数:
		combined_list (list): 原始列表。
		sample_size (int): 指定的采样数量。

	返回:
		list: 包含 sample_size 个元素的随机采样列表。
	"""
	# 如果原列表为空，直接返回空列表
	if not combined_list:
		return []

	# 如果原列表的长度大于或等于指定的采样数量，使用 random.sample
	if len(combined_list) >= sample_size:
		return random.sample(combined_list, sample_size)
	# 如果原列表的长度小于指定的采样数量，使用 random.choices 上采样
	else:
		return random.choices(combined_list, k=sample_size)

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


def select_smiles_for_combination(df,combo_types):
	"""
	根据组合类型，随机选择SMILES
	"""
	selected_smiles = []
	for type_ in combo_types:
		# 筛选出 polymer_type 等于当前类型的行
		filtered_df = df[df['polymer_type'] == type_]
		
		if filtered_df.empty:
			return f"No matching polymer_type found for: {type_}"
		
		# 随机选择一行
		random_row = filtered_df.sample(n=1)
		
		# 获取 smiles
		selected_smiles.append(random_row['smiles'].values[0])
	
	return selected_smiles



def combine_lists(A, B, C):
	"""
	穷举出 bratch, single, double 中元素的所有组合，并将每个组合的子列表合并成一个单一的子列表

	:param A: 列表 A
	:param B: 列表 B
	:param C: 列表 C
	:return: 包含所有合并后子列表的列表
	"""
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


# ### 根据组合类型生成新数据

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


# In[11]:


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
	edcombo = Chem.EditableMol(combo)
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


# In[14]:


def get_neiid_bysymbol(combo,symbol):
	for at in combo.GetAtoms():
		if at.GetSymbol()==symbol:
			at_nei=at.GetNeighbors()
			return at_nei.GetIdx()


# In[15]:


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


# In[16]:


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


# In[17]:


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


# In[18]:


def remove_atom_labels_from_smiles(smiles):
	# 解析SMILES字符串，生成一个分子对象
	mol = Chem.MolFromSmiles(smiles)
	
	
	# 遍历分子中的所有原子，去掉原子标记
	for atom in mol.GetAtoms():
		atom.SetAtomMapNum(0)  # 设置原子标记为0，相当于去掉标记
 
	new_smiles = Chem.MolToSmiles(mol)
	return new_smiles


# In[19]:


def remove_all_parentheses_C_from_smiles(smiles):
	# 使用正则表达式一次性替换所有 "(C)"
	cleaned_smiles = re.sub(r'\(C\)', '', smiles)
	return cleaned_smiles


# In[20]:


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


# In[21]:


def molecular_to_polymer(smi):
	mol = Chem.MolFromSmiles(smi)
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



def join_with_underscore(lst):
	# Filter out empty strings and None values
	filtered_list = [str(item) for item in lst if item not in (None, '')]
	# Join the elements with underscores
	return '_'.join(filtered_list)

def process_combination(combination,ring_df, max_smi=1000,replace=True):
	connection_num =100
	com_name = join_with_underscore(combination)
	new_smi_list = []
	fragments = select_smiles_for_combination(ring_df, combination)
	all_connections = generate_all_possible_connections(fragments)
	if len(all_connections)>0:
		all_connections = random_sample(all_connections,connection_num)
	else:
		return [],[]
	count = 0
	for connection in all_connections:
		connection = list(connection)
		connection, lists = rename_mark(connection)
		pairs = generate_combinations(lists)
	  
		if len(pairs) == len(connection) - 1:
			new_str = combinefragbydifferentmatrix(connection, pairs)
			if new_str:
				mol = Chem.MolFromSmiles(new_str)
				if replace:
					smi = replace_patterns(new_str)
				else:
					smi = new_str
				if smi:
					mol2 = Chem.MolFromSmiles(smi)
					if mol and mol2:
						smi = molecular_to_polymer(smi)
						if smi:
							mol = Chem.MolFromSmiles(smi)
							if mol:
								new_smi_list.append(smi)
								# 如果达到上限，提前返回结果
								if max_smi is not None and len(new_smi_list) >= max_smi:
									return new_smi_list, [com_name for _ in range(len(new_smi_list))]
	
	# 返回所有可能的结果
	return new_smi_list, [com_name for _ in range(len(new_smi_list))]  
  
def save_to_csv(data, file_name):
	df = pd.DataFrame(data, columns=['smiles'])
	df.to_csv(file_name, index=False)


def parallel_main(combined_list, ring_df, output_file, num_workers=1, max_smi=1000, replace=True):
	total_smi_list = []
	total_com_list = []
	
	with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
		futures = []
		for combination in combined_list:
			futures.append(executor.submit(process_combination, combination, ring_df, max_smi, replace))
		count = 0
		for future in concurrent.futures.as_completed(futures):
			count += 1
			smi_list, com_list = future.result()
			if smi_list:
				logging.info(f"Number of results in set{count}: {len(smi_list)}")
				total_smi_list += smi_list
				total_com_list += com_list
				del smi_list, com_list	# 可选：立即释放临时变量
			gc.collect()  # 强制回收（尤其在内存敏感时）
	
	# 随机采样逻辑
	if data_num is not None and len(total_smi_list) > data_num:
		random_indices = random.sample(range(len(total_smi_list)), data_num)
		sampled_smi_list = [total_smi_list[i] for i in random_indices]
		sampled_com_list = [total_com_list[i] for i in random_indices]
		del total_smi_list, total_com_list	# 释放原始大列表
	else:
		sampled_smi_list, sampled_com_list = total_smi_list, total_com_list
	
	df = pd.DataFrame({'smiles': sampled_smi_list, 'combination': sampled_com_list})
	df.to_csv(output_file, index=False)
	del sampled_smi_list, sampled_com_list	# 可选：释放采样后的列表
	gc.collect()  # 最终清理
	logging.info(f"Saved result to {output_file}")	
	print("Processing complete.")

ring_df,index_frame=get_com.get_ring_df(input_file,smiles_index=smiles_index,label_index=label_index,threshold=threshold,filter_direction=filter_direction,sample_size=sample_size)
combined_list= get_com.get_combination(ring_df,index_frame)

total_new_list = []
com_list = []
count = 0


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#random_combinations = random_sample(combined_list, type_num)
parallel_main(combined_list,ring_df,output_file,num_workers,max_smi,replace)
#for combination in random_combinations:
#	 results, com = process_combination(combination,ring_df,max_smi=max_smi,replace_patterns=False)
#	 logging.info(f"Processing result set: {count}")
#	 count += 1
#	 total_new_list += results
#
#if len(total_new_list) > data_num:
#	 smi_list = random.sample(total_new_list, data_num)
#else:
#	 smi_list = total_new_list
#df = pd.DataFrame({'smiles': smi_list})
#df.to_csv(output_file, index=False)
#logging.info(f"Number of results in this set: {len(results)}")
#logging.info(f"Saved result to {output_file}")
#	  
#print("Processing complete.")


