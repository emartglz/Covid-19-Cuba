#By @Kike_XD

from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.optimize import curve_fit
import numpy as np
import json
import requests
from pprint import pprint
from matplotlib.animation import FuncAnimation
import pandas as pd
import datetime
import IPython.utils.colorable as colorr

provinces = ["Pinar del Río", "Artemisa", "La Habana", "Mayabeque", "Matanzas", "Cienfuegos", "Ciego de Ávila", "Villa Clara", "Sancti Spíritus", "Camagüey", "Las Tunas", "Granma", "Guantánamo", "Santiago de Cuba", "Holguín", "Isla de la Juventud"]
municipies = ['Diez de Octubre', 'Guanabacoa', 'Playa', 'Plaza de la Revolución', 'Boyeros', 'Centro Habana', 'Cotorro', 'Arroyo Naranjo', 'La Lisa', 'La Habana Vieja', 'San Miguel del Padrón', 'Cerro', 'Habana del Este', 'Marianao', 'Regla']

def plot_saved_deads_found_per_day(dates, cases, deads, saves):
    plt.figure(figsize=(20, 7), dpi=80)
    plt.plot(cases, label = 'casos informados')
    plt.fill_between(range(len(dates)), cases, facecolor='blue', alpha=0.5)
    plt.plot(deads, label = 'casos fallecidos', color='red')
    plt.fill_between(range(len(dates)), deads, facecolor='red', alpha=0.5)
    plt.plot(saves, label = 'casos recuperados', color='green')
    plt.fill_between(range(len(dates)), saves, facecolor='green', alpha=0.5)
    plt.xticks(range(len(cases)), dates, rotation=90, size=8)
    plt.legend()
    plt.savefig('cases,_deads,_saves_per_day.svg')

def plot_cases_per_day(dates, cases):
    plt.figure(figsize=(20, 7), dpi=80)
    plt.plot(cases, marker='o', label = 'casos informados')
    plt.ylabel('cantidad de casos infectados por día')
    plt.xticks(range(len(cases)), dates, rotation=90, size=8)
    plt.legend()
    plt.savefig('cases_per_day.svg')
    # plt.show()

def plot_active_cases(dates, cases, deads, saves):
    x = np.arange(len(dates))
    y = np.array(cases)

    for i in range(1, len(y)):
        y[i] += y[i -1]
        y[i] -= deads[i]
        y[i] -= saves[i]

    plt.figure(figsize=(20, 7), dpi=80)
    plt.plot(y, marker='o', label = 'casos informados')
    plt.ylabel('cantidad de casos infectados activos')
    plt.xticks(range(len(cases)), dates, rotation=90, size=8)
    plt.legend()
    plt.savefig('active_cases.svg')

def plot_dead_cases(dates, deads):
    plt.figure(figsize=(20, 7), dpi=80)
    plt.plot(deads, marker='o', label = 'casos informados', color='red')
    plt.ylabel('cantidad de casos fallecidos por día')
    plt.xticks(range(len(deads)), dates, rotation=90, size=8)
    plt.legend()
    plt.savefig('dead_per_day.svg')

def plot_safed_cases(dates, saved):
    plt.figure(figsize=(20, 7), dpi=80)
    plt.plot(saved, marker='o', label = 'casos informados', color='green')
    plt.ylabel('cantidad de casos recuperados por día')
    plt.xticks(range(len(saved)), dates, rotation=90, size=8)
    plt.legend()
    plt.savefig('saved_per_day.svg')

def plot_acumulado(dates, cases, deads, saves):
    plt.figure(figsize=(20, 7), dpi=80)

    y = np.cumsum(cases)
    plt.plot(y, marker='o', label = 'casos totales', color = 'blue')
    y = np.cumsum(saves)
    plt.plot(y, marker='o', label = 'recuperados totales', color = 'green')
    y = np.cumsum(deads)
    plt.plot(y, marker='o', label = 'muertes totales', color = 'red')
    plt.xticks(range(len(cases)), dates, rotation=90, size=8)
    plt.legend()
    plt.savefig('total_cases.svg')
    # plt.show()

def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

def plot_aprox_exp(dates, cases, deads, saves):
    x = np.arange(len(dates))
    y = np.array(cases)

    for i in range(1, len(y)):
        y[i] += y[i -1]
        y[i] -= deads[i]
        y[i] -= saves[i]

    popt, pcov = curve_fit(exp_func, x, y)

    plt.figure(figsize=(20, 7), dpi=80)
    plt.plot(y, marker='o', label = 'casos informados')
    plt.plot(x, exp_func(x, *popt))
    x2 = np.array([x[len(x) - 1], x[len(x) - 1] + 1, x[len(x) - 1] + 2])
    # print(*popt)
    plt.plot(x2, exp_func(x2, *popt), 'ro--', label = 'pronóstico de casos para los próximos días')
    plt.xticks(range(len(cases)), dates, rotation=90, size=8)
    plt.ylabel('cantidad de casos infectados totales')
    plt.legend()
    plt.savefig('aprox_exp.svg')
    # plt.show()

def plot_poly_aprox(dates, cases, deads, saves):
    x = np.arange(len(dates))
    y = np.array(cases)

    for i in range(1, len(y)):
        y[i] += y[i -1]
        y[i] -= deads[i]
        y[i] -= saves[i]
    
    p = np.poly1d(np.polyfit(x, y, 3))

    plt.figure(figsize=(20, 7), dpi=80)
    plt.plot(y, marker='o', label = 'casos activos')
    plt.plot(x, p(x))
    x2_ = []
    for i in range(10):
        x2_.append(x[len(x) - 1] + i)
    x2 = np.array(x2_)
    plt.plot(x2, p(x2), 'ro--', label = 'pronóstico de casos para los próximos días')
    plt.xticks(range(len(cases)), dates, rotation=90, size=5)
    plt.ylabel('cantidad de casos infectados totales')
    plt.legend()
    plt.ylim(bottom = 0)
    plt.savefig('aprox_poly.svg')
    # plt.show()

def plot_diag_cerro(days):
    dates = []
    cases_cerro = []
    cases_hav = []

    for i in range(len(days)):
        dates.append(days[str(i + 1)]['fecha'])
        try:
            temp = len(days[str(i + 1)]['diagnosticados'])
            cases_hav.append(0)
            cases_cerro.append(0)
            for j in range(temp):
                try:
                    if(days[str(i + 1)]['diagnosticados'][j]['provincia_detección'] == provinces[2]):
                        cases_hav[i] += 1
                    if(days[str(i + 1)]['diagnosticados'][j]['municipio_detección'] == "Cerro"):
                        cases_cerro[i] += 1
                except:
                    pass
        except:
            cases_cerro.append(0)
            cases_hav.append(0)

    plt.figure(figsize=(20, 7), dpi=80)
    plt.plot(cases_hav, marker='o', label = 'casos informados en La Habana')
    plt.plot(cases_cerro, marker='o', label = 'casos informados en el Cerro')
    plt.xticks(range(len(cases_hav)), dates, rotation=90, size=8)
    plt.legend()
    plt.savefig('cases_per_day_cerro.svg')

def plot_diag_havana(days):
    dates = []
    cases = []
    cases_hav = []

    for i in range(len(days)):
        dates.append(days[str(i + 1)]['fecha'])
        try:
            cases.append(len(days[str(i + 1)]['diagnosticados']))
            cases_hav.append(0)
            for j in range(cases[i]):
                try:
                    if(days[str(i + 1)]['diagnosticados'][j]['provincia_detección'] == provinces[2]):
                        cases_hav[i] += 1
                except:
                    pass
        except:
            cases.append(0)
            cases_hav.append(0)

    plt.figure(figsize=(20, 7), dpi=80)
    plt.plot(cases, marker='o', label = 'casos informados en Cuba')
    plt.plot(cases_hav, marker='o', label = 'casos informados en La Habana')
    plt.xticks(range(len(cases)), dates, rotation=90, size=8)
    plt.legend()
    plt.savefig('cases_per_day_hav.svg')

def find(array, x):
    for i, p in enumerate(array):
        if p == x:
            return i

def daytime_trans(str):
    p = str.split('/')
    for i, num in enumerate(p):
        p[i] = int(num)
    return datetime.datetime(p[0], p[1], p[2])

def nice_axes(ax):
    ax.set_facecolor('.8')
    ax.tick_params(labelsize = 8, length = 0)
    ax.grid(True, axis='x', color='white')
    ax.set_axisbelow(True)
    [spine.set_visible(False) for spine in ax.spines.values()]

def prepare_data(df, steps=5):
    df = df.reset_index()
    df.index = df.index * steps
    
    last_idx = df.index[-1] + 1
    df_expanded = df.reindex(range(last_idx))

    df_expanded['index'] = df_expanded['index'].fillna(method='ffill')
    df_expanded = df_expanded.set_index('index')
    df_rank_expanded = df_expanded.rank(axis=1, method='first')

    df_expanded = df_expanded.interpolate()
    df_rank_expanded = df_rank_expanded.interpolate()

    return df_expanded, df_rank_expanded

def rgb_tp_prgb(r, g, b):
    temp = 1 / 255
    return [r * temp, g * temp, b * temp, 1]

colors = [rgb_tp_prgb(8, 84, 57),rgb_tp_prgb(179, 0, 225),rgb_tp_prgb(29, 98, 171),rgb_tp_prgb(114, 180, 222),rgb_tp_prgb(6, 168, 65),rgb_tp_prgb(255, 0, 153),rgb_tp_prgb(248, 199, 28), rgb_tp_prgb(203, 74, 10),rgb_tp_prgb(243, 145, 0),rgb_tp_prgb(79, 0, 4),rgb_tp_prgb(0, 155, 83),rgb_tp_prgb(218, 37, 28),rgb_tp_prgb(17, 19, 20),rgb_tp_prgb(254, 50, 0),rgb_tp_prgb(0, 141, 217),rgb_tp_prgb(0, 165, 138)]
    #provinces = ["Pinar del Río",   "Artemisa",            "La Habana",             "Mayabeque",                 "Matanzas",            "Cienfuegos",              "Ciego de Ávila",          "Villa Clara",              "Sancti Spíritus",     "Camagüey",            "Las Tunas",           "Granma",                "Guantánamo",          "Santiago de Cuba",    "Holguín",              "Isla de la Juventud"]


def animate_cases_provinces(days):
    dates = []
    cases = [[0 for j in range(len(provinces))] for i in range(len(days))]

    for i in range(len(days)):
        
        dates.append(daytime_trans(days[str(i + 1)]['fecha']))
        try:
            l = len(days[str(i + 1)]['diagnosticados'])

            for j in range(l):
                try:
                    cases[i][find(provinces, days[str(i + 1)]['diagnosticados'][j]['provincia_detección'])] += 1
                except:
                    pass
        except:
            pass
        if i > 0:
            for p in range(len(provinces)):
                cases[i][p] += cases[i-1][p]
        
    df = pd.DataFrame(cases, dates, provinces)
    df_expanded, df_rank_expanded = prepare_data(df, 25)

    labels = df_expanded.columns


    fig = plt.figure(figsize=(16, 8), dpi=144)
    ax = fig.subplots()

    def init():
        ax.clear()
        nice_axes(ax)

    def update(i):
        init()
        for bar in ax.containers:
            bar.remove()
        y = df_rank_expanded.iloc[i]
        width = df_expanded.iloc[i]
        ax.barh(y=y, width=width, color=colors, tick_label=labels, edgecolor=[1, 1, 1, 1])
        for j in range(len(y)):
            ax.text(width[j], y[j] - 0.15, str(int(width[j])), fontsize='smaller')
        date_str = df_expanded.index[i].strftime('%B%-d, %Y')
        ax.set_title(f'COVID-19 Total cases by province - {date_str}', fontsize='smaller')

    anim = FuncAnimation(fig=fig, func=update, init_func=init, frames=len(df_expanded), interval=1000/30, repeat=False)
    anim.save('covid19 por provincias.mp4')
    # plt.show()


def animate_cases_havana(days):
    dates = []
    cases = [[0 for j in range(len(municipies))] for i in range(len(days))]

    for i in range(len(days)):
        
        dates.append(daytime_trans(days[str(i + 1)]['fecha']))
        try:
            l = len(days[str(i + 1)]['diagnosticados'])

            for j in range(l):
                try:
                    if days[str(i + 1)]['diagnosticados'][j]['provincia_detección'] == 'La Habana':
                        cases[i][find(municipies, days[str(i + 1)]['diagnosticados'][j]['municipio_detección'])] += 1
                except:
                    pass
        except:
            pass
        if i > 0:
            for p in range(len(municipies)):
                cases[i][p] += cases[i-1][p]
        
    df = pd.DataFrame(cases, dates, municipies)
    df_expanded, df_rank_expanded = prepare_data(df, 25)

    labels = df_expanded.columns

    fig = plt.figure(figsize=(16, 8), dpi=144)
    ax = fig.subplots()

    def init():
        ax.clear()
        ax.set_xlim([0, 1])
        nice_axes(ax)

    def update(i):
        init()
        for bar in ax.containers:
            bar.remove()
        y = df_rank_expanded.iloc[i]
        width = df_expanded.iloc[i]
        ax.barh(y=y, width=width, color=colors, tick_label=labels, edgecolor=[1, 1, 1, 1])
        for j in range(len(y)):
            ax.text(width[j], y[j] - 0.15, str(int(width[j])), fontsize='smaller')
        date_str = df_expanded.index[i].strftime('%B%-d, %Y')
        ax.set_title(f'COVID-19 Total cases by municipies in Havana - {date_str}', fontsize='smaller')

    anim = FuncAnimation(fig=fig, func=update, init_func=init, frames=len(df_expanded), interval=1000/30, repeat=False)
    anim.save('covid19 por municipio en La Habana.mp4')
    # plt.show()

def read_json(p):
    with open(p) as fp:
        return json.load(fp)

def read_days(data):
    return data['casos']['dias']

def read_date_and_cases(days):
    l = len(days)

    date = []
    cases = []
    deads = []
    saves = []

    for i in range(l):
        date.append(days[str(i + 1)]['fecha'])
        try:
            cases.append(len(days[str(i + 1)]['diagnosticados']))
        except KeyError:
            cases.append(0)
        try:
            deads.append(days[str(i + 1)]['muertes_numero'])
        except KeyError:
            deads.append(0)
        try:
            saves.append(days[str(i + 1)]['recuperados_numero'])
        except KeyError:
            saves.append(0)

    return (date, cases, deads, saves)

def seed_data(url):
    res = requests.get(url)
    
    with open('data.json', 'w') as the_file:
        the_file.writelines(res.text)

def main():
    url = 'https://covid19cubadata.github.io/data/covid19-cuba.json'

    seed_data(url)
    data = read_json('data.json')

    assert data != None, 'ERROR READING FILE'

    days = read_days(data)
    date, cases, deads, saves = read_date_and_cases(days)
    
    print("WORKING")
    plot_diag_havana(days)
    plot_diag_cerro(days)
    plot_acumulado(date, cases, deads, saves)
    # plot_aprox_exp(date, cases, deads, saves)
    # plot_poly_aprox(date, cases, deads, saves)
    plot_active_cases(date, cases, deads, saves)
    plot_cases_per_day(date, cases)
    plot_dead_cases(date, deads)
    plot_safed_cases(date, saves)
    plot_saved_deads_found_per_day(date, cases, deads, saves)
    # animate_cases_provinces(days)
    animate_cases_havana(days)

if __name__ == '__main__':
    main()