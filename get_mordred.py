from rdkit import Chem
from mordred import Calculator, descriptors
import	csv, os
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, ChemicalFeatures
from sklearn import preprocessing
import pickle as pkl


def read_data(file_name,smile_index):
	#导入数据
	df = pd.read_csv(file_name)
	if type(smile_index) == str:
		smi_list0 = df[smile_index].tolist()
	elif type(smile_index) == int:
		smi_list0 = df.iloc[:, smile_index]
	smi_list=[]
	name_list=[]
	mol_list=[]
	for idx,i in enumerate(smi_list0):
		#Rdkit识别不出顺反，要把'/'与'\'去掉
		#if '/'in i[1]:
			#i[1]=i[1].replace('/','')
		#if '\\'in i[1]:
			#i[1]=i[1].replace('\\','')
		if '()' in i:
			i = i.replace('()','(C)')
		mol=Chem.MolFromSmiles(i)
		if mol:
			try:
				AllChem.EmbedMolecule(mol)
			except:
				continue
			smi=Chem.MolToSmiles(mol)
			smi_list.append(smi)
			name_list.append(idx)
			mol_list.append(mol)
	return smi_list,mol_list,name_list


def get_mordred(mol_list,name_list):
	#建立Mordred描述符
	calc = Calculator(descriptors, ignore_3D=False)
	df = calc.pandas(mol_list)
	df.index=name_list
	return df


def data_processing(df,zscore_scaler=False,scaler_feature=False):
	#数据标准化
	if zscore_scaler == False:
		df = df._get_numeric_data()
		zscore_scaler = preprocessing.StandardScaler() 
		result= zscore_scaler.fit_transform(df)	 
	elif scaler_feature != False:
		df = df[scaler_feature]
		result= zscore_scaler.transform(df)
	result = pd.DataFrame(result,index = df.index,columns = df.columns)
	scaler_feature = result.columns.tolist()
	return result,zscore_scaler,scaler_feature

def get_model(file_name):
	with open(file_name, "rb") as file:
		target_name, model, selected_feature, importance_df ,zscore_scaler,scaler_feature= pkl.load(file)
	return model,selected_feature,zscore_scaler,target_name,scaler_feature

def get_mordred_from_feature_list(mol_list,name_list,feature_list,Mordred_df):
	calc_descriptors = Mordred_df[Mordred_df['feature_index'].isin(feature_list)]['descriptors'].values.tolist()
	calc = Calculator(calc_descriptors,ignore_3D=False)
	df = calc.pandas(mol_list)
	df = df.applymap(lambda x: x if isinstance(x, (int, float)) else 0)
	df.index=name_list
	return df

def get_data(file_name, selected_feature, smile_index,Mordred_df,zscore_scaler,scaler_feature):
	smi_list, mol_list, name_list = read_data(file_name, smile_index=smile_index)
	mordred = get_mordred(mol_list,name_list)
	mordred ,zscore_scaler,scaler_feature = data_processing(mordred,zscore_scaler,scaler_feature)
	mordred = mordred[selected_feature]
	return mordred,name_list

def get_label(file_name,name_list,target_name):
	#获取标签
	df_t = pd.read_csv(file_name)
	df_t = df_t.loc[name_list]#确保标签和特征值一致
	y = np.array(df_t[target_name])
	return y

def get_predict(model_data, predict_data,Mordred_df,label_Flag=False,smile_index=0):
	model,selected_feature,zscore_scaler,target_name,scaler_feature = get_model(model_data)
	X,name_list = get_data(file_name,selected_feature,smile_index,Mordred_df,zscore_scaler,scaler_feature)
	predictions = model.predict(X)
	label = []
	if label_Flag == True:
		label = get_label(predict_data,name_list,target_name)
	return predictions,label

def get_mordred_df():
	#创建Mordred索引
	feature_index = []
	cal = Calculator(descriptors, ignore_3D=False)
	for d in cal.descriptors:
		feature_index.append(str(d))
		
	descriptors = [d for d in cal.descriptors]
	# 创建 DataFrame
	Mordred_df = pd.DataFrame({'feature_index': feature_index, 'descriptors': descriptors})
	return Mordred_df

def get_X_y(file_name,selected_feature, Mordred_df,target_name,smile_index,zscore_scaler,scaler_feature):
	X,name_list = get_data(file_name, selected_feature, smile_index,Mordred_df,zscore_scaler,scaler_feature)
	y = get_label(file_name,name_list,target_name)
	
	return X,y

