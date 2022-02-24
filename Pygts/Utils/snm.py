# coding=utf-8
import pandas as pd
import sys

sys.path.append('/home/pyserver_dev')
import db_utils as db_utils
import joblib
import warnings
import datetime
from tqdm import tqdm

warnings.filterwarnings('ignore')
import pickle

import datetime
import pandas as pd
import numpy as np
import datetime as dt
from scipy.fft import fft
from scipy import signal
import multiprocessing.dummy as mp
import scipy.signal as sg
from datetime import datetime, timedelta

import earthtide
__all__ = ['calculate_julian_century', 'solve_longman_tide', 'solve_longman_tide_scalar',
           'solve_tide_df', 'solve_point_corr']


def get_yesterday():
    # 获取今天（现在时间）
    today = datetime.today()
    yesterday = today - timedelta(days=1)
    return str(yesterday.year) + '-' + str(yesterday.month) + '-' + str(yesterday.day)


def get_solid(lat, lon, start_time, end_time, step_sec, lock):
    '''
        获取固体潮
        ----------
        :param lat: 站台维度,float 34.0
        :param lon: 站台经度,float -118.0
        :param start_time: 开始时间,str 2019-03-17 11:00:00
        :param end_time: 结束时间,str 2020-03-17 11:00:00
        :param step_sec: 时间间隔,int 300
        :return tide_u: 固体潮,list [1,3,4]
    '''
    # lock.acquire()
    # # 转换时间
    # (dt_out, tide_e, tide_n, tide_u) = pysolid.calc_solid_earth_tides_point(lat, lon, start_time, end_time,
    #                                                                         step_sec=step_sec,
    #                                                                         display=False,
    #                                                                         verbose=False)
    # lock.release()
    # # 暂时返回 向上的固体潮
    # solid_result = np.array(tide_u) * 100
    # dump_pickle(solid_result, '10月昆明固体潮')
    s = (end_time - start_time).days * 24 * 3600 + (end_time - start_time).seconds + 1
    year, month, day, hour, miniute, second = [], [], [], [], [], []
    for i in range(s):
        tmp_time = start_time + timedelta(seconds=i)
        year.append(tmp_time.year)
        month.append(tmp_time.month)
        day.append(tmp_time.day)
        hour.append(tmp_time.hour)
        miniute.append(tmp_time.minute)
        second.append(tmp_time.second + 1)

    solid_result = earthtide.TheoryGravityTide(lon, lat, 0, 1.16, year, month, day, hour, miniute, second)

    return solid_result


def get_polynome(g):
    '''
        九阶多项式
        ----------
        :param g: 重力数据,list [5012,5201,5125]
        :return y: 九阶多项式后的数据
    '''
    # 生成x等差数列
    x = np.linspace(0, len(g), len(g))
    # 九阶多项式拟合
    b = np.polyfit(x, g, 9)
    # 获取多项式公式
    c = np.poly1d(b)
    # 返回重新计算结果
    polynome_result = np.array(c(x))
    return polynome_result


def prepare_data(tag, origin_data_df, origin_solid_dict, lock):
    '''
        背景噪声处理数据准备
        ----------
        :param origin_data_df: 原数据读取表
        :param origin_solid_dict: 固体潮数据
        :param step_sec: 时间间隔
        :return result_data: 处理后的重力数据,以降采频率做key
    '''
    # 初始化结果
    result = {}
    q_list = list(set(origin_data_df['downsample_q']))

    # 遍历每一个采样点
    for q in q_list:
        origin_data_group = origin_data_df[origin_data_df['downsample_q'] == q]
        # 分割数据
        date_time = origin_data_group.iloc[0]['date_time']  # 时间
        g = np.array(np.array(origin_data_group['g'])[0]).astype('float')  # 重力数据
        p = np.array(np.array(origin_data_group['p'])[0]).astype('float')  # 气压数据

        # 获取固体潮
        solid = origin_solid_dict[q][:int(86400 / q)]  # 当前固体潮
        dump_pickle(solid, str(tag) + '_固体潮')
        surplus_solid = origin_solid_dict[q][int(86400 / q):]  # 剩余固体潮
        origin_solid_dict[q] = surplus_solid  # 替换信值

        # 扣除固体潮与大气压后的值
        y_tmp = g - solid + 0.3 * p
        dump_pickle(y_tmp, str(tag) + '_扣除固体潮与大气压后的值')

        # 九阶多项式拟合
        polynome = get_polynome(y_tmp)
        dump_pickle(polynome, str(tag) + '_九阶多项式的拟合结果')

        # 扣除大气压固体潮
        result[q] = y_tmp - polynome
        dump_pickle(y_tmp - polynome, str(tag) + '_扣除九阶多项式结果')
    return result


def transform_data(tag, g_result, delta=1):
    '''
        算法计算转换
        ----------
        :param g_result: 重力处理后的数据,list [1,2,3]
        :return snm_dict: 台站每日的snm
    '''
    snm_list = []

    # 遍历日均psd
    for k, v in g_result.items():
        new_v = []
        mean = np.mean(v)
        v = v - mean
        # # 求hann
        window = signal.hann(len(v))
        for i in range(len(v)):
            new_v.append(v[i] * window[i] * ((1 / 0.375) ** 0.5))
        dump_pickle(new_v, str(tag) + '_加窗结果')

        origin_signal_len = len(new_v)
        new_signal_len = 0
        n = 0
        for i in range(1, 21):
            if 2 ** i > len(new_v):
                n = i
                new_signal_len = 2 ** i
                break
            else:
                continue

        # 零填充
        zero_list = np.zeros(new_signal_len - origin_signal_len)
        new_v = np.append(new_v, zero_list)

        ############################# 自己复现的 #############################
        # # 傅里叶变换
        # nyq = int(new_signal_len/2 + 1)
        # g_fft = fft(new_v, n=new_signal_len)
        # sumfe = 0
        # for i in range(nyq):
        #     delta = 2.0
        #     if i ==0 or i==nyq-1:
        #         delta = 1.0
        #     sumfe += g_fft[i] * np.conj(g_fft[i]) * delta
        # sumfe = sumfe*delta/new_signal_len
        # psd_mean = sumfe/(86400*delta)
        ############################# 自己复现的 #############################

        ############################# 使用PSD公式 1/600 - 1/200 #############################
        fs, psd = signal.periodogram(new_v, nfft=new_signal_len)
        sumfe = []
        for i in range(len(fs)):
            if fs[i] < 1 / 200 and fs[i] > 1 / 600:
                sumfe.append(psd[i])
        psd_mean = np.mean(sumfe)
        ############################# 使用PSD公式 1/600 - 1/200 #############################

        snm = np.log10(psd_mean) + 2.5
        dump_pickle(snm, str(tag) + '_snm结果')

        # 添加结果
        snm_list.append({
            'downsample_q': k,
            'mean_psd': psd_mean,
            'snm': snm
        })
    # 返回snm算法
    return snm_list


def fill_na(data_list):
    '''
    空值填充
    :param data_list: 输入序列
    :return:
    '''
    new_data_list = []
    for i in data_list:
        if i == 'NULL':
            new_data_list.append(np.nan)
        else:
            new_data_list.append(float(i))

    # 空值填充
    data_df = pd.DataFrame(new_data_list, columns=['tmp_col'])
    data_df['tmp_col'] = data_df['tmp_col'].fillna(data_df['tmp_col'].interpolate())
    # 使用均值再次填充
    data_df['tmp_col'].fillna(data_df['tmp_col'].mean(), inplace=True)
    return data_df['tmp_col'].tolist()


def dowmsample(tag, origin_data_df, code_p, code_g, q_list, n=8, ftype='iir', zero_phase=True):
    '''
    数据降采样
    :param origin_data_df: 原始数据的一天，包含重力与大气压两笔
    :param q: 采样频率
    :param n: 滤波器阶数
    :param ftype: 滤波器类型
    :param zero_phase: 0相位
    :return: result_df: 采样后的数据，列名：时间-站点-采样频率-重力-大气压
    '''
    result_dict = []
    try:
        p_x = origin_data_df[origin_data_df['ITEMID'] == code_p].iloc[0]['OBSVALUE']  # 获取大气压值
        p_x = p_x.rstrip()
        p_x = p_x.split(' ')
    except Exception as ex:
        raise Exception('缺少大气压数据')

    try:
        g_x = origin_data_df[origin_data_df['ITEMID'] == code_g].iloc[0]['OBSVALUE']  # 重力值
        g_x = g_x.rstrip()
        g_x = g_x.split(' ')
    except Exception as ex:
        raise Exception('缺少重力值数据')

    # 如果出现缺失值，忽略该次计算
    if len(p_x) != 86400 or len(g_x) != 86400:
        raise Exception('重力值或大气压数据长度不足86400')

    p_x = fill_na(p_x)
    g_x = fill_na(g_x)
    for q in q_list:
        q = int(q)
        # 初始化结果
        g_data_downsample = g_x.copy()
        p_data_downsample = p_x.copy()

        # 若采样频率不为1，证明需要进行采样，否则直接采用元数据
        if q != 1:
            # 构建采样段数
            sample_q_list = None
            if q <= 60:  # 1分钟内采样
                sample_q_list = [60]
            elif q > 60 and q <= 300:  # 5分钟内采样
                sample_q_list = [60, int(q / 60)]
                if q % 60 != 0:
                    sample_q_list.append(q % 60)
            elif q > 300 and q <= 3600:  # 1小时内采样
                sample_q_list = [60, 5, int(q / 300)]
                if q % 300 != 0:
                    sample_q_list.append(q % 300)
            else:  # 1小时外采样
                sample_q_list = [60, 5, 12, int(q / 3600)]
                if q % 3600 != 0:
                    sample_q_list.append(q % 3600)

            # 遍历开始分段采样
            for sample_q in sample_q_list:
                if len(g_data_downsample) >= 30:
                    g_data_downsample = sg.decimate(g_data_downsample, sample_q, n=n, ftype=ftype,
                                                    zero_phase=zero_phase)
                    p_data_downsample = sg.decimate(p_data_downsample, sample_q, n=n, ftype=ftype,
                                                    zero_phase=zero_phase)
        # 记录结果
        result_dict.append({
            'date_time': origin_data_df.iloc[0]['STARTDATE'],
            'stationid': origin_data_df.iloc[0]['STATIONID'],
            'downsample_q': q,
            'g': g_data_downsample,
            'p': p_data_downsample
        })
    # 返回dataframe
    dowmsample_result = pd.DataFrame(result_dict)
    return dowmsample_result


lock = mp.Manager().Lock()

error_list = []
result_list = []
origin_data_df = pd.read_csv('data/无标题_2022-01-07.csv')

origin_data_df = origin_data_df[origin_data_df['STATIONID'] == 'YNKM1']
# 获取固体潮数据
solid_start_time = datetime.strptime('2020-10-01 00:00:00', '%Y-%m-%d %H:%M:%S')
solid_end_time = datetime.strptime('2020-10-31 23:59:59', '%Y-%m-%d %H:%M:%S')
origin_solid_dict = {}
for i in tqdm([1]):
    origin_solid_dict[int(i)] = get_solid(25.14, 102.74, solid_start_time,
                                          solid_end_time, int(i), lock)

# 对时间进行分组遍历
origin_data_df['STARTDATE'] = pd.to_datetime(origin_data_df['STARTDATE'])
origin_data_df = origin_data_df.sort_values(by=['STARTDATE'], axis=0, ascending=True)  # 时间升序
origin_data_df['year'] = origin_data_df['STARTDATE'].dt.year
origin_data_df['month'] = origin_data_df['STARTDATE'].dt.month
origin_data_df['day'] = origin_data_df['STARTDATE'].dt.day
for date, origin_data_group in tqdm(origin_data_df.groupby(['year', 'month', 'day'])):
    tag = str(date[0]) + '-' + str(date[1]) + '-' + str(date[2])
    # 数据降采样
    try:
        downsample_data_df = dowmsample(tag, origin_data_group, 2128, 2121, [1])
    except Exception as ex:
        error_list.append(
            {'station': 'YNKM1', 'error_sql': 'data_query', 'error_step': '数据缺失',
             'error_msg': str(date) + ': ' + str(ex)})
        continue
    # 数据准备
    g_result = prepare_data(tag, downsample_data_df, origin_solid_dict, lock)
    # 计算算法
    snm_list = transform_data(tag, g_result)
    for i in snm_list:
        i['network_code'] = 'YNKM1'
        i['date'] = origin_data_group.iloc[0]['STARTDATE']
        # TODO 保存数据
        result_list.append(i)
result_df = pd.DataFrame(result_list)
