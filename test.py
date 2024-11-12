from Enviroment import Enviroment
import plotly.graph_objects as go
import torch
import os
import numpy as np
from datetime import datetime


def test(args, epoch, agent, val_mem, metrics, results_dir, evaluate=False):
    env = Enviroment(args,eval=True) #evaluation enviroment
    metrics['steps'].append(epoch)
    test_rewards, test_q = [] ,[]

    #Test the performance over several episodes
    done = True
    for _ in range(args.eval_episodes):
        while True:
            if done or truncated: #resetting the enviroment
                state, _ = env.reset()
                r_sum = 0
                done = False
            
            state = torch.tensor(np.array(state, dtype=np.float32), dtype=torch.float32, device=args.device).div_(255) #the input for the network is a torch tensor
            action = agent.select_action(state, greedy=True)
            state, reward, done, truncated, _ = env.step(action)
            r_sum += reward
            
            if done or truncated: #checks if the number of frames in the episode has exceeded the limit or the episode is done
                test_rewards.append(r_sum)
                break
    env.close()

    #Test Q-values over validation memory
    for state in val_mem:
        test_q.append(agent.evaluate_q(state))
    
    avg_reward = sum(test_rewards) / len(test_rewards)
    avg_q = sum(test_q) / len(test_q)

    if not evaluate:
        #new model test if its better
        if avg_reward > metrics['best_avg_reward']:
            metrics['best_avg_reward'] = avg_reward
            agent.save(results_dir)

        # Append to results and save metrics
        metrics['rewards'].append(test_rewards)
        metrics['Qs'].append(test_q)
        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

        # Plot
        _plot_line(metrics['steps'], metrics['rewards'], 'Rewards', path=results_dir)
        _plot_line(metrics['steps'], metrics['Qs'], 'Qs', path=results_dir)

    _log('Epoch=' + str(epoch) + '/' + str(args.max_epoch) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_q))
    return avg_reward, avg_q

# Simple ISO 8601 timestamped logger
def _log(s):
  print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)

#Plot q-step and learning charts
def _plot_line(xs, ys, title, path=''):
    # Calculate min, max, and mean rewards
    mins = np.min(ys, axis=1)
    maxs = np.max(ys, axis=1)
    means = np.mean(ys, axis=1)

    # Create traces
    trace_mean = go.Scatter(x=xs, y=means, mode='lines', name=('Mean ' + title), line=dict(color='rgba(221,68,17,255)'))
    trace_min_max = go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([mins, maxs[::-1]]),
        fill='toself',
        fillcolor='rgba(238,161,136,255)',  # Transparent color for shaded region
        line=dict(color='rgba(0,0,0,0)'),  # Transparent line color
        showlegend=False
    )

    # Create layout
    layout = go.Layout(title= (title + ' Variation Over Epochs'), xaxis=dict(title='training steps'), yaxis=dict(title=title),  plot_bgcolor='rgba(0,0,0,0)')

    # Create figure
    fig = go.Figure(data=[trace_min_max, trace_mean], layout=layout)

    # Save plot to HTML file
    fig.write_html(os.path.join(path, title + '.html'))


