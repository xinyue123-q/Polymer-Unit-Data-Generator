import numpy as np
import pandas as pd
import csv, os
from collections import Counter
from rdkit import Chem
from rdkit.Chem import AllChem
#from rdkit.Chem.Draw import IPythonConsole 
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions 
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import Draw
import structure_identity_tool as F
from IPython.display import display
from rdkit.Chem import Draw
import math
from PIL import Image,ImageDraw, ImageFont
import os
import matplotlib.pyplot as plt
import pickle as pkl
import random
def random_sample_rows(file_name, sample_size):
	"""
	该函数用于从指定文件读取数据框，并随机选取指定数量的行。

	参数:
	file_name (str): 要读取的 CSV 文件的文件名。
	sample_size (int): 需要随机选取的行数。

	返回:
	pandas.DataFrame: 包含随机选取行的数据框。
	"""
	# 读取 CSV 文件为数据框
	df = pd.read_csv(file_name)
	# 如果数据框的行数小于要采样的数量，直接返回原数据框
	if len(df) <= sample_size:
		return df
	# 随机选取 sample_size 个不重复的行索引
	random_indices = random.sample(range(len(df)), sample_size)
	# 根据随机选取的索引获取对应行的数据
	sampled_df = df.iloc[random_indices]
	return sampled_df

def processing_data(file_name,smile_index=0,label_index=None,sample_size=None):
	#导入数据
	
	df = pd.read_csv(file_name)
	if sample_size:
		df = random_sample_rows(file_name, sample_size)
	if type(smile_index) == str:
		smi_list0 = df[smile_index].tolist()
	elif type(smile_index) == int:
		smi_list0 = df.iloc[:, smile_index]
	if label_index :
		if type(label_index) == str:
			label_list0 = df[label_index].tolist()
		elif type(label_index) == int:
			label_list0 = df.iloc[:, label_index]
	else:
		label_list0 = [None for i in range(len(smi_list0))]
		
	smi_list=[]
	name_list=[]
	mol_list=[]
	label_list=[]
	
	for idx,i in enumerate(smi_list0):
		#Rdkit识别不出顺反，要把'/'与'\'去掉
		if '"'in i:
			i=i.strip('"')
		#if '\\'in i[1]:
			#i[1]=i[1].replace('\\','')
		mol=Chem.MolFromSmiles(i)		 
		if mol:
			smi=Chem.MolToSmiles(mol)
			label = label_list0[idx]
			label_list.append(label)
			smi_list.append(smi)
			name_list.append(idx)
	return smi_list,name_list,label_list


def get_pu(smi_list,name_list):
	ring_total_list=[]
	error_find_independent_str1=[]
	total_neighbor_data={}
	n=0
	while n < len(smi_list):
	
		smiles=smi_list[n]
		name=name_list[n] 
		
		from_get_bracket_index=F.get_bracket_index(smiles)
		left_index_list=from_get_bracket_index[0]
		right_index_list=from_get_bracket_index[1]
		index_list=from_get_bracket_index[2]
		
	
		cp_list=F.pairing(smiles,index_list,left_index_list,right_index_list)
		cp_arr=np.array(cp_list)
		index_arr=np.array(index_list)
		
		smallest_r=F.smallest(cp_list,index_arr)
			
		str_df=F.structure_DataFrame(cp_list,smallest_r,right_index_list,left_index_list)
	
		independent_cp_and_dependent_cp=F.rigin_type_classify(cp_list,smiles,smallest_r,str_df)
		independent_cp=independent_cp_and_dependent_cp[0]
		dependent_cp=independent_cp_and_dependent_cp[1]
		bratch=independent_cp_and_dependent_cp[3]
		bratch_cp=independent_cp_and_dependent_cp[2]
	
		cp_data=F.get_cp_data(cp_list,smallest_r,str_df,independent_cp,bratch_cp)
	
		find_str=F.find_independent_str(smiles,smallest_r,cp_data,independent_cp,dependent_cp,bratch_cp)
		string0=find_str[0]
		index_data=find_str[1]
		index_cp=find_str[2]
		index_data0=find_str[3]
	   
		br={}
		index_data2={}
		for k,v in index_data.items():
			j=v[1]
			j2=F.add_bracket(j)
			j3=F.make_smi(j2)
			j4=F.link_c(j3)
			br_f=F.bratch_in_string(j4)
			j5=br_f[0]
			bratch1=br_f[1]
			mol=Chem.MolFromSmiles(j5)
			if mol:
				 j6= Chem.MolToSmiles(mol)
			br_f2=F.bratch_in_string(j6)
			j7=br_f2[0]
			bratch2=bratch1+br_f2[1]
			br[k]=bratch2
			v2=[v[0],j7]
			index_data2[k]=v2
		make_con_data=F.make_con(index_data2,index_cp,br)
		index_data3=make_con_data[0]
		#index_data4=F.delete_free_radical_in_index_data(index_data3)
		index_cp2=make_con_data[1]
		br2=make_con_data[2]
		br3=F.bratch_amend(br2)
		for k,v in index_data3.items():
			ring_total_list.append(v[1])
		for k,v in br3.items():
			v2=[]
			for j in v:
				mol=Chem.MolFromSmiles(j)
				if mol:
					smi=Chem.MolToSmiles(mol)
					v2.append(smi)
					ring_total_list.append(smi)
			br3[k]=v2
		neighbor_data=F.found_neighbor(br3,str_df,index_data3,index_cp2)
		neighbor_data2=F.found_end_point_neighbour(smiles,neighbor_data,index_data3) 
	  
		#for k,v in neighbor_data2.items():
		#	for k2,v2 in v['right_neighbor'].items():
		#		if '[C]'in v2:
		#			v2=v2.replace('[C]','C')
		#	for k2,v2 in v['left_neighbor'].items():
		#		if '[C]'in v2:
		#			v2=v2.replace('[C]','C')
		#	if '[C]'in v['self']:
		#		v['self']=v['self'].replace('[C]','C')
		total_neighbor_data[name]=neighbor_data2
		n=n+1
		
	ring_total_list.sort()
	ring_total_list2=[]
	for i in set(ring_total_list):
		ring_total_list2.append(i)
	#ring_arr=np.array(ring_total_list2)
	#ring_df=pd.DataFrame(ring_arr)
	#ring_df.to_csv('output/ring_total_list.csv')
	return ring_total_list2,total_neighbor_data	  
		

def get_img(ring_total_list2):
	# 确保保存图像的目录存在
	output_dir = 'output/img'
	os.makedirs(output_dir, exist_ok=True)
	
	# 确保保存拼接后大图的目录存在
	output_dir2 = 'output/img2'
	os.makedirs(output_dir2, exist_ok=True)
	
	# 遍历每个 SMILES 字符串并保存为图像
	for idx, smiles in enumerate(ring_total_list2):
		mol = Chem.MolFromSmiles(smiles)
		if mol:
			fig = Draw.MolToImage(mol, size=(900, 900), kekulize=True)
			fig.save(os.path.join(output_dir, f'{idx + 1}.png'))
		
	print(f'All images saved to {output_dir}')
	
	# 获取所有小图片的路径
	image_folder = 'output/img'
	small_images = os.listdir(image_folder)
	
	# 自定义排序函数，提取文件名中的数字并将其转换为整数
	def get_number(filename):
		return int(filename[:-4])
	
	# 使用 sorted 函数按数字从小到大排序
	small_images = sorted(small_images, key=get_number)
	
	# 确定大图片的尺寸
	num_images = len(small_images)
	row = 3	 # 每行放置的小图片数量
	column = 4	# 每列放置的小图片数量
	num_per_pic = row * column	# 每张大图的小图数量
	row_height = 900  # 小图片的高度
	row_width = 900	 # 小图片的宽度
	big_width = row * row_width	 # 大图片的宽度
	big_height = column * row_height  # 大图片的高度
	
	# 创建一个空白图案
	blank_image = Image.new("RGB", (row_width, row_height), "white")
	
	# 计算需要空白图片的数量
	remainder = num_images % num_per_pic
	blank_count = num_per_pic - remainder
	
	# 对于能够拼接成整张大图的部分
	num_images2 = num_images - remainder  # 能拼成整张大图的小图片数量
	count = 0
	for i in range(0, num_images2, num_per_pic):
		# 创建一个新的大图片对象
		big_image = Image.new('RGB', (big_width, big_height))
		# 遍历每张小图片并拼接到大图片上
		for j, image_name in enumerate(small_images[i:i + num_per_pic]):
			image_path = os.path.join(image_folder, image_name)
			small_image = Image.open(image_path)
			# 创建绘制对象
			draw = ImageDraw.Draw(small_image)
			# 定义文本标签的样式
			font = ImageFont.truetype("arial.ttf", size=35)
			# 获取文本的宽度和高度
			text = 'polymer_unit:' + image_name[:-4]
			text_width, text_height = draw.textsize(text, font=font)
			# 计算居中的位置
			position = ((row_width - text_width) // 2, 0)
			# 绘制文本标签
			draw.text(position, text, font=font, fill=(0, 0, 0))
			# 计算小图片的坐标
			x = (j % row) * row_width
			y = (j // row) * row_height
			# 将小图片粘贴到大图片上
			big_image.paste(small_image, (x, y))
		img_path = os.path.join(output_dir2, f'image{count}.jpg')
		count += 1
		# 保存拼接好的大图片
		big_image.save(img_path)
	
	# 拼接剩余的图片和空白图片
	big_image = Image.new('RGB', (big_width, big_height))
	for i in range(num_per_pic):
		if i < remainder:
			num = num_images2 + i
			image_name = small_images[num]
			image_path = os.path.join(image_folder, image_name)
			small_image = Image.open(image_path)
			# 创建绘制对象
			draw = ImageDraw.Draw(small_image)
			# 定义文本标签的样式
			font = ImageFont.truetype("arial.ttf", size=40)
			# 获取文本的宽度和高度
			text = 'polymer_unit:' + image_name[:-4]
			text_width, text_height = draw.textsize(text, font=font)
			# 计算居中的位置
			position = ((row_width - text_width) // 2, 1)
			# 绘制文本标签
			draw.text(position, text, font=font, fill=(0, 0, 0))
			# 计算小图片的坐标
			x = (i % row) * row_width
			y = (i // row) * row_height
			# 将小图片粘贴到大图片上
			big_image.paste(small_image, (x, y))
		else:
			# 计算小图片的坐标
			x = (i % row) * row_width
			y = (i // row) * row_height
			# 将空白图片粘贴到大图片上
			big_image.paste(blank_image, (x, y))
	
	# 保存拼接后的大图
	img_path = os.path.join(output_dir2, f'image{count}.jpg')
	big_image.save(img_path)  # 可以根据需要自定义文件名和格式
	
	print(f'All big images saved to {output_dir2}')



def get_one_hot(ring_total_list2,total_neighbor_data,name_list):
	
	ring_series=pd.Series(ring_total_list2)
	m=0
	for k,v in total_neighbor_data.items():
		if len(v)>m:
			m=len(v)
	long=len(ring_series)
	fp_list=[]
	for i in name_list:
		fp1=np.zeros((1,long))
		data=total_neighbor_data[i]
		for k,v in data.items():
			self=v['self']
			column=ring_series[ring_series.values==self].index[0]
			fp1[:,column]=int(1)
		fp1=fp1.tolist()[0]
		fp_list.append(fp1)
	
	fp_arr=np.array(fp_list)
	fp_df=pd.DataFrame(fp_arr,index=name_list)
	
	#fp_df.to_csv("output/one_hot.csv")
	return fp_df



def get_number(ring_total_list2,total_neighbor_data,name_list):
	ring_series=pd.Series(ring_total_list2)
	
	m=0
	for k,v in total_neighbor_data.items():
		if len(v)>m:
			m=len(v)
	long=len(ring_series)
	fp_list=[]
	for i in name_list:
	
		fp1=np.zeros((1,long))
		data=total_neighbor_data[i]
		for k,v in data.items():
			self=v['self']
			column=ring_series[ring_series.values==self].index[0]
			fp1[:,column]=fp1[:,column]+int(1)
		fp1=fp1.tolist()[0]
		fp_list.append(fp1)
		
	
	fp_arr=np.array(fp_list)
	fp_df=pd.DataFrame(fp_arr,index=name_list)
	
	#fp_df.to_csv("output/number.csv")
	return fp_df


def get_index_list(name_list,ring_total_list2,total_neighbor_data):
	ring_series=pd.Series(ring_total_list2)
	
	m=0
	for k,v in total_neighbor_data.items():
		if len(v)>m:
			m=len(v)
	long=len(ring_series)
	total_index_list=[]
	for i in name_list:
		
		data=total_neighbor_data[i]
		fp=np.full((1,m),'none')[0]
		index_list=[]
		for k,v in data.items():
			self=v['self']
			#print(f"self:{self}")
			#print(f"ring_series[ring_series.values==self]:{ring_series[ring_series.values==self]}")
			index=ring_series[ring_series.values==self].index[0]
			index_list.append(index)
		n=len(index_list)
		j=0
		while j<n:
			fp[j]=index_list[j]
			j=j+1
		total_index_list.append(fp)
	index_frame=pd.DataFrame(total_index_list,index=name_list)
	#index_frame.to_csv('output/index_data.csv')
	return index_frame


def one_hot(file_name,index):
	smi_list,name_list,_ = processing_data(file_name,index)
	ring_total_list2,total_neighbor_data = get_pu(smi_list,name_list)
	one_hot = get_one_hot(ring_total_list2,total_neighbor_data,name_list)
	return one_hot

def number(file_name,index):
	smi_list,name_list,_ = processing_data(file_name,index)
	ring_total_list2,total_neighbor_data = get_pu(smi_list,name_list)
	number = get_number(ring_total_list2,total_neighbor_data,name_list)
	return number

if __name__ == "__main__":
	smi_list,name_list,_ = processing_data('input/zinc.csv')
	ring_total_list2,total_neighbor_data = get_pu(smi_list,name_list)
	ring_df = pd.DataFrame([])
	ring_df['smiles'] = ring_total_list2
	ring_df.to_csv('output/zinc_ring.csv')
	with open('output/total_neighbor_data.pkl', 'wb') as f:
		pkl.dump(total_neighbor_data, f)
	with open('output/total_neighbor_data.pkl', 'rb') as f:	 
		total_neighbor_data = pkl.load(f)
	df = pd.read_csv('output/zinc_ring.csv')
	ring_total_list2 = df['smiles'].tolist()
	index_frame=get_index_list(name_list,ring_total_list2,total_neighbor_data)
	#get_one_hot(ring_total_list2,total_neighbor_data)
	#get_number(ring_total_list2,total_neighbor_data)
	#get_index_list(ring_total_list2,total_neighbor_data)
	#get_img(ring_total_list2)

