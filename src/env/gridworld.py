import numpy as np
import sys
from gym.envs.toy_text import discrete
from contextlib import closing
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# define the actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridworldEnv(discrete.DiscreteEnv):
    """
    A 4x4 Grid World environment from Sutton's Reinforcement 
    Learning book chapter 4. Termial states are top left and
    the bottom right corner.

    Actions are (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave agent in current state.
    Reward of -1 at each step until agent reachs a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, size = (4,4), rewards = None, terminal_states = None, transition_cost = 1):
        self.shape = size
        self.nS = np.prod(self.shape)
        self.nA = 4
        self.transition_cost = transition_cost
        if rewards is None:
            rewards = np.zeros(self.nS)
            rewards[0] = 0
            rewards[-1] = 0
        self.rewards = rewards
        if terminal_states is None:
            terminal_states = [0, self.nS - 1]
        self.terminal_states = terminal_states

        P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(self.nA)}
            P[s][UP] = self._transition_prob(position, [-1, 0])
            P[s][RIGHT] = self._transition_prob(position, [0, 1])
            P[s][DOWN] = self._transition_prob(position, [1, 0])
            P[s][LEFT] = self._transition_prob(position, [0, -1])

        # Initial state distribution is uniform
        isd = np.ones(self.nS) / self.nS

        # We expose the model of the environment for dynamic programming
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridworldEnv, self).__init__(self.nS, self.nA, P, isd)

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _transition_prob(self, current, delta):
        """
        Model Transitions. Prob is always 1.0.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition
        :return: [(1.0, new_state, reward, done)]
        """

        # if stuck in terminal state
        current_state = np.ravel_multi_index(tuple(current), self.shape)
        if current_state in self.terminal_states:
            return [(1.0, current_state, self.rewards[current_state], True)]\

        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)

        is_done = new_state in self.terminal_states
        return [(1.0, new_state, self.rewards[new_state] - self.transition_cost, is_done)]

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            # Print terminal state
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += '\n'

            outfile.write(output)
        outfile.write('\n')

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
            
    # Custom print to show state values inside the grid
    def grid_print(self, V, k=None):
        ax = sns.heatmap(V.reshape(self.shape),
                        annot=True, square=True,
                        cbar=False, cmap='RdYlGn',
                        xticklabels=False, yticklabels=False)

        if k:
            ax.set(title="State values after K = {0}".format(k))
        else:
            ax.set(title="State Values".format(k))
        plt.show()

class Video_callback:
    heatmap_params = dict(colorscale = 'RdYlGn',
                            texttemplate = '%{z}',
                            showlegend=False,
                            showscale=False,
                            )
    
    def __init__(self, shape, value_range = None,):
        
        self.frames = []
        self.titles = []
        if value_range is not None:
            self.heatmap_params['zmin'] = value_range[0]
            self.heatmap_params['zmax'] = value_range[1]
        self.shape = shape

    def write(self, V, k=None):
        # Format array so that it render collectly
        self.frames.append(np.flip(V.reshape(self.shape), axis = 0))
        self.titles.append(k)

    def plot(self):
        fig_dict = {
            "data": [],
            "layout": {
                            'xaxis': {'visible': False},
                            'yaxis': {'visible': False,
                                        'scaleanchor': 'x'}},
            "frames": []
        }
        fig_dict["layout"]["updatemenus"] = [
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True, "transition": {"duration": 300,
                                                                            "easing": "quadratic-in-out"}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }
        ]
        sliders_dict = {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Iteration: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": []
        }
        fig_dict['data'] = go.Frame(data=go.Heatmap(z=self.frames[0], **self.heatmap_params))['data']

        for i, arr in enumerate(self.frames):
            
            fig_dict["frames"].append(go.Frame(data=go.Heatmap(z=arr, **self.heatmap_params), name = str(i)))

            slider_step = {"args": [
                [i],
                {"frame": {"duration": 300, "redraw": True},
                "mode": "immediate",
                "transition": {"duration": 300}
                }
            ],
                "label": str(i),
                "method": "animate"}
            sliders_dict["steps"].append(slider_step)
        fig_dict["layout"]["sliders"] = [sliders_dict]
        fig = go.Figure(fig_dict)
        fig.update_traces(showlegend=False)
        fig.update(layout_showlegend=False)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        return fig
            
