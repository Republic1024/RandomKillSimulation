import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import warnings
import matplotlib as mpl
import seaborn as sns
from pylab import mpl, plt
# 设置警告忽略和样式
warnings.filterwarnings('ignore')
sns.set_style("white")
mpl.rcParams['font.family'] = '微软雅黑'


def run_simulation_numpy(num_people):
    """
    执行一次模拟游戏，返回每个人被淘汰的轮次。
    """
    ids = np.arange(1, num_people + 1)
    killed_turn = np.full(num_people, -1)  # 初始化淘汰回合
    turn_num = num_people

    for turn in range(turn_num):
        alive_ids = ids[killed_turn == -1]  # 获取当前存活的ID
        if len(alive_ids) == 0:  # 如果没有存活者，提前退出循环
            break

        # 选择一个随机的奇数索引
        random_odd_idx = random.randrange(1, len(alive_ids) + 1, 2)
        kill_id = alive_ids[random_odd_idx - 1] - 1  # 选择要淘汰的ID

        killed_turn[kill_id] = turn + 1  # 记录淘汰回合

    return killed_turn, kill_id + 1


# 模拟参数
num_people = 600
N = 200000

# 初始化DataFrame
df_simul = pd.DataFrame({"id": range(1, 1 + num_people)})

# 预先分配足够的内存空间
np_simul = np.zeros((num_people, N + 1), dtype=int)
np_simul[:, 0] = np.arange(1, 1 + num_people)

winner_ids = []
# 运行模拟
for i in tqdm(range(N)):
    # winner id add to list and val count
    result_np, winner_id = run_simulation_numpy(num_people)
    np_simul[:, i + 1] = result_np
    winner_ids.append(winner_id)

# 转换为DataFrame
df_simul = pd.DataFrame(np_simul)

a = winner_ids
b = np.array(range(1, 1 + num_people))

last_killed_counts = pd.DataFrame({
    'id':
    b,
    'win_count':
    pd.DataFrame(np.concatenate((a, b))).value_counts().sort_index().values - 1
})
last_killed_counts.plot(
    x='id',
    y='win_count',
    figsize=(13, 6),
    title=
    f"Each id win counts: {N} turns for {num_people} people kill simulation")

plt.savefig(
    f"./result/Each id win counts: {N} turns for {num_people} people kill simulation.svg"
)
last_killed_counts.to_csv("./result/stats/each_ids_win_counts.csv",
                          index=False)

df_simul['average_alive_turns'] = df_simul.iloc[:, 1:].mean(axis=1)
df_simul['id'] = np.array(range(1, 1 + num_people))
df_simul[['id',
          'average_alive_turns']].to_csv("./result/stats/each_ids_avg_survival_turns.csv",
                          index=False)

df_simul[['id', 'average_alive_turns']].plot(
    x='id',
    y='average_alive_turns',
    figsize=(13, 6),
    title=
    f"Each id average survival turns: {N} turns for {num_people} people kill simulation"
)
plt.savefig(
    f"./result/Each id average survival turns: {N} turns for {num_people} people kill simulation.svg"
)
