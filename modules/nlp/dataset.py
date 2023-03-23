import datetime
import random
import gc

import numpy as np
import pandas as pd
import torch


class SessionDataset(torch.utils.data.IterableDataset):

    def __init__(self, sessions, user_column='user_id', event_prefix='event', time_prefix='time', padding_idx=0,
                 negative_samples=15, time_unit=datetime.timedelta(seconds=1)):
        self.negative_samples = negative_samples
        self.time_unit = time_unit
        self.padding_idx = padding_idx
        self.event_columns = [col for col in sessions.columns if col.startswith(event_prefix)]
        self.time_columns = [col for col in sessions.columns if col.startswith(time_prefix)]
        # map session to events with time
        self.users = sessions[user_column].unique().tolist()
        events = pd.Series(zip(*(sessions[f'{col}'].tolist() for col in self.event_columns)))
        events_times = pd.Series(zip(*(sessions[f'{col}'].tolist() for col in self.time_columns)))

        data = sessions.loc[:, [user_column]]
        data.loc[:, event_prefix] = events
        data.loc[:, time_prefix] = events_times
        user_data = data.groupby(user_column)
        data = {user: user_data.get_group(user) for user in self.users}
        self.data = {
            user:
                [{
                    'events': evs,
                    'times': tms
                } for evs, tms in zip(data[user][event_prefix].tolist(), data[user][time_prefix].tolist())]
            for user in self.users}
        del events, events_times, data, user_data
        gc.collect()

    def __iter__(self):
        return self

    def __next__(self):
        base_user = random.choice(self.users)
        neg_users = []
        while len(neg_users) < self.negative_samples:
            neg_user = random.choice(self.users)
            if neg_user == base_user:
                continue
            neg_users.append(neg_user)

        items = [random.choice(self.data[user]) for user in [base_user] + neg_users]

        events, times = [], []
        for item in items:

            events.append(item['events'])

            times.append([0])
            event_times = item['times']
            for time_id in range(1, len(event_times)):
                times[-1].append((event_times[time_id] - event_times[time_id - 1]) // self.time_unit)

        return {
            'users': np.array(base_user, dtype=np.int64),
            'events': np.array(events, dtype=np.int64),  # (1 + neg) x t
            'times': np.array(times, dtype=np.float32)
        }
