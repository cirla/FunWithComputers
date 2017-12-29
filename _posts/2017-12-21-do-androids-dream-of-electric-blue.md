---
layout: post
title:  Do Androids Dream of Electric Blue?
date:   2017-12-21 20:00:00 -0500
---

Paint colors always have such fanciful names like "Flamingo's Dream" and "Agreeable Gray".
Can we teach a computer to invent new colors and give them fitting names? Let's give it a shot!

## Corpus Colorum

First, let's gather some paint color information from existing brands.

Digging through the sources on color explorers for [Benjamin Moore](http://www.benjaminmoore.com/en-us/for-your-home/color-gallery), [Sherwin-Williams](http://www.sherwin-williams.com/homeowners/color/find-and-explore-colors/paint-colors-by-family/), and [Behr](http://www.behr.com/consumer/colors/paint), I was able to find some JSON endpoints that could give us names, RGB values, and color families for all of their currently available colors. There's some other information (e.g. "color collection", "goes great with") that might be fun to play around with, but for now we'll just grab this simple information, e.g.:

```python
{
    'name': 'SYLVAN MIST',
    'rgb': (184, 199, 191),
    'family': 'BLUE',
}
```

If data gathering and cleaning sounds boring to you, skip ahead to [Exploring the Corpus](#exploring-the-corpus).

We'll define a few utility functions as we go along to help us homogenize the data.

```python
ALLOWED_SPECIAL_CHARS = set(' \'"-.,&?')
def clean_name(n):
    """Strip superfluous characters and convert to uppercase"""
    return ''.join([
        c for c in n.upper()
        if c.isalnum() or c in ALLOWED_SPECIAL_CHARS])
```

The Benjamin Moore API gives us data pretty close to what we need.
We'll just have to convert the RGB values from hexadecimal and clean up the names.


```python
import struct

import pandas as pd
import requests

def rgb_from_hex(h):
    """Convert hex string (e.g. "00FF00") to tuple of RGB values (e.g. (0, 255, 0))"""
    return struct.unpack('BBB', bytes.fromhex(h))

def load_benjamin_moore():
    data = requests.get('https://www.benjaminmoore.com/api/colors').json()
    df = pd.DataFrame(list(data['colors'].values()),
                      columns=['name', 'family', 'hex'])
    df[['name', 'family']] = df[['name', 'family']].apply(lambda x: x.apply(clean_name))
    df['rgb'] = df['hex'].apply(rgb_from_hex)

    return df.drop('hex', axis=1)

benjamin_moore_colors = load_benjamin_moore()
len(benjamin_moore_colors)
```

    4221

```python
benjamin_moore_colors.sample()
```

|| name | family | rgb |
|-|-:|-:|-:|
| **3378** | WORN LEATHER SHOES | NEUTRAL | (152, 142, 120) |

Sherwin Williams is also close, but for some reason the RGB values are given as a single integer.

```python
def rgb_from_dec(d):
    """Convert integer (e.g. 65280) to tuple of RGB values (e.g. (0, 255, 0))"""
    return rgb_from_hex(f'{d:06x}')

def load_sherwin_williams():
    data = requests.get('https://www.sherwin-williams.com/color-visualization/services/color/SW/all').json()
    df = pd.DataFrame(data, columns=['name', 'colorFamilyNames', 'rgb'])

    df['name'] = df['name'].apply(clean_name)
    df['family'] = df['colorFamilyNames'].apply(lambda x: clean_name(x[0]))
    df['rgb'] = df['rgb'].apply(rgb_from_dec)

    return df.drop('colorFamilyNames', axis=1)

sherwin_williams_colors = load_sherwin_williams()
len(sherwin_williams_colors)
```

    1746

```python
sherwin_williams_colors.sample()
```

|| name | family | rgb |
|-|-:|-:|-:|
| **190** | AMBITIOUS AMBER | ORANGE | (240, 203, 151) |

Behr is a bit tricky as the data is inside of a JavaScript source file instead of a JSON endpoint.
Also, the color family data is stored separately from the color information, so we'll have to join the two together.

```python
import itertools

def get_data_list(js_source):
    """
    Extract the JSON string representing a list from a JavaScript
    source file of the form 'var data = [ ... ];'
    """
    return js_source[js_source.find('[') : js_source.rfind(']') + 1]

def flatten_groups(groups):
    """
    For some reason, groups are stored as a list of lists of strings
    (which are themselves comma-separated lists of color IDs). Flatten
    this into a single list of unique color IDs.

    For example, [['a,b,c', 'd,e'], ['f,g', 'h,i,j'], ['k', 'a,b']]
    would become ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
    """
    return list(set(
        itertools.chain.from_iterable(
            x.split(',')
            for x in itertools.chain.from_iterable(groups))))

def load_behr():
    colors_data = requests.get('http://www.behr.com/mainService/services/colornx/all.js').text
    df = pd.read_json(get_data_list(colors_data))

    # extract first row as column names
    df.columns = df.iloc[0]
    df = df.reindex(df.index.drop(0))[['id', 'name', 'rgb']]

    family_data = requests.get('http://www.behr.com/mainService/services/xml/families.js').text
    family_df = pd.read_json(get_data_list(family_data))
    family_df['groups'] = family_df['groups'].apply(flatten_groups)

    # explode `groups` column into a column for each value in the list
    # e.g. {'name': 'Red', 'groups': ['a', 'b', 'c']}
    # becomes {'name': 'Red', '0': 'a', '1': 'b', '2': 'c']}
    family_df = pd.concat([family_df['name'],
                           family_df['groups'].apply(pd.Series)],
                          axis=1)

    # melt group columns into a single column, creating a row from each
    # e.g. {'name': 'Red', '0': 'a', '1': 'b', '2': 'c']}
    # becomes {'name': 'Red', 'id': 'a'}, {'name': 'Red', 'id': 'b'}, {'name': 'Red', 'id': 'c'}
    family_df = pd.melt(family_df,
                        id_vars=['name'],
                        value_name='id')[['name', 'id']].dropna()

    # join families to colors by ID
    df = df.merge(family_df, on='id', suffixes=['_color', '_family'])

    df[['name', 'family']] = df[['name_color', 'name_family']].apply(lambda x: x.apply(clean_name))
    df['rgb'] = df['rgb'].apply(lambda x: rgb_from_hex(x[1:]))

    return df[['name', 'family', 'rgb']]

behr_colors = load_behr()
len(behr_colors)
```

    2891

```python
behr_colors.sample()
```

|| name | family | rgb |
|-|-:|-:|-:|
| **1650** | DRIED CHAMOMILE | YELLOW | (209, 179, 117) |

Now that they're all in the same format, we can combine them all together.

```python
colors = pd.concat([benjamin_moore_colors,
                    sherwin_williams_colors,
                    behr_colors]).drop_duplicates()
len(colors)
```

    8137

There are a bunch of colors with weird family names like 'Timeless Color' or 'Historic Color'.
For simplicity, let's discard these.

```python
COLOR_FAMILIES = set([
    'RED', 'ORANGE', 'PINK', 'BROWN', 'NEUTRAL', 'GRAY',
    'WHITE', 'YELLOW', 'PURPLE', 'BLUE', 'BLACK', 'GREEN'])

colors = colors[
    colors['family'].isin(COLOR_FAMILIES)]

len(colors)
```

    7405

<a id='exploring-the-corpus'></a>
### Exploring the Corpus

It would be neat if we could view the colors inline.

```python
from IPython.display import HTML

def display_color(color):
    return HTML("""
        <div style="width: 128px; display: inline-block">
            <p><div style="font-weight: bold">{name}</div>{family}</p>
            <p><svg width="64" height="64" style="background: #{rgb_hex}" /></p>
            <p>#{rgb_hex}</p>
        </div>
    """.format(
        name=color.name,
        family=color.family,
        rgb_hex=struct.pack('BBB', *color.rgb).hex()))

def display_colors(colors_df):
    return HTML("""
        <div>
            {colors}
        </div>
    """.format(
        colors = '\n'.join(
            display_color(c).data for c in colors_df.itertuples())))
```

```python
display_colors(colors.sample(5))
```

<div>
  <div style="width: 128px; display: inline-block">
    <p><div style="font-weight: bold">EMERGENCY ZONE</div>ORANGE</p>
    <p><svg width="64" height="64" style="background: #e36841" /></p>
    <p>#e36841</p>
  </div>
  <div style="width: 128px; display: inline-block">
    <p><div style="font-weight: bold">PAR FOUR</div>GREEN</p>
    <p><svg width="64" height="64" style="background: #d2d6c7" /></p>
    <p>#d2d6c7</p>
  </div>
  <div style="width: 128px; display: inline-block">
    <p><div style="font-weight: bold">HACIENDA BLUE</div>BLUE</p>
    <p><svg width="64" height="64" style="background: #0087a9" /></p>
    <p>#0087a9</p>
  </div>
  <div style="width: 128px; display: inline-block">
    <p><div style="font-weight: bold">HIGHLAND THISTLE</div>RED</p>
    <p><svg width="64" height="64" style="background: #b9a0b0" /></p>
    <p>#b9a0b0</p>
  </div>
  <div style="width: 128px; display: inline-block">
    <p><div style="font-weight: bold">PASSION VINE</div>GREEN</p>
    <p><svg width="64" height="64" style="background: #888169" /></p>
    <p>#888169</p>
  </div>
</div>

Let's see which families have the most shades:

```python
colors.groupby('family').count() \
      .rename(columns={'name': 'count'}) \
      .sort_values('count', ascending=False) \
      .reset_index()[['family', 'count']]
```

<div>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>family</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GREEN</td>
      <td>1145</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BLUE</td>
      <td>995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RED</td>
      <td>938</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ORANGE</td>
      <td>862</td>
    </tr>
    <tr>
      <th>4</th>
      <td>YELLOW</td>
      <td>766</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BROWN</td>
      <td>755</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PURPLE</td>
      <td>650</td>
    </tr>
    <tr>
      <th>7</th>
      <td>GRAY</td>
      <td>600</td>
    </tr>
    <tr>
      <th>8</th>
      <td>WHITE</td>
      <td>339</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NEUTRAL</td>
      <td>287</td>
    </tr>
    <tr>
      <th>10</th>
      <td>BLACK</td>
      <td>51</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PINK</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>

Who knew there could be 51 Shades of Black?

```python
display_colors(colors[colors['family'] == 'BLACK'].sample(5))
```

<div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">TWILIGHT ZONE</div>BLACK</p>
      <p><svg width="64" height="64" style="background: #2f3234" /></p>
      <p>#2f3234</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">DEEP CAVIAR</div>BLACK</p>
      <p><svg width="64" height="64" style="background: #453f3f" /></p>
      <p>#453f3f</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">CHEATING HEART</div>BLACK</p>
      <p><svg width="64" height="64" style="background: #494c4d" /></p>
      <p>#494c4d</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">BLACK SATIN</div>BLACK</p>
      <p><svg width="64" height="64" style="background: #2a2d2e" /></p>
      <p>#2a2d2e</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">ABYSS</div>BLACK</p>
      <p><svg width="64" height="64" style="background: #3f4348" /></p>
      <p>#3f4348</p>
  </div>
</div>

There are some colors with more than one name. Let's take a look at the one with the most names.


```python
most_common_rgb = colors[colors.duplicated(['rgb'], keep=False)] \
                        .groupby('rgb').count() \
                        .rename(columns={'name': 'count'}) \
                        .nlargest(1, ['count']) \
                        .reset_index().iloc[0]['rgb']
display_colors(colors[colors['rgb'] == most_common_rgb])
```

<div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">WHITE SWAN</div>NEUTRAL</p>
      <p><svg width="64" height="64" style="background: #ebe4d0" /></p>
      <p>#ebe4d0</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">SPANISH WHITE</div>WHITE</p>
      <p><svg width="64" height="64" style="background: #ebe4d0" /></p>
      <p>#ebe4d0</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">GRAND TETON WHITE</div>NEUTRAL</p>
      <p><svg width="64" height="64" style="background: #ebe4d0" /></p>
      <p>#ebe4d0</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">BATTENBERG</div>NEUTRAL</p>
      <p><svg width="64" height="64" style="background: #ebe4d0" /></p>
      <p>#ebe4d0</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">MONTEREY WHITE</div>NEUTRAL</p>
      <p><svg width="64" height="64" style="background: #ebe4d0" /></p>
      <p>#ebe4d0</p>
  </div>
</div>

## All in the Family

A good start for assigning a name to a random color would be to first figure out to which family it belongs.

We're going to try a few different classifiers, so let's wrap them in a similar interface:

```python
from sklearn.utils import shuffle

class ColorFamilyClassifier:
    def __init__(self, color_df, train_percent=0.8):
        self.shuffled = shuffle(color_df)

        data = [
            (self.get_features(color), self.get_label(color))
            for color in self.shuffled.itertuples()
        ]

        cut_index = round(train_percent * len(data))
        self.train_set = data[:cut_index]
        self.test_set = data[cut_index:]

        self.init_classifier()

    def get_features(self, color):
        raise NotImplemented

    def get_label(self, color):
        raise NotImplemented

    def init_classifier(self):
        raise NotImplemented

    def accuracy(self):
        raise NotImplemented

    def classify(self, color):
        raise NotImplemented
```

### Naïve Bayes

Let's start by seeing how much mileage we can get with a [Naïve Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) using RGB values as features.

```python
import nltk

class NaiveBayesClassifier(ColorFamilyClassifier):
    def get_label(self, color):
        return color.family

    def init_classifier(self):
        self.classifier = nltk.NaiveBayesClassifier.train(self.train_set)

    def accuracy(self):
        return nltk.classify.accuracy(self.classifier, self.test_set)

    def classify(self, color):
        return self.classifier.classify(self.get_features(color))

class NaiveBayesRGBClassifier(NaiveBayesClassifier):
    def get_features(self, color):
        return dict(zip(("red", "green", "blue"), color.rgb))
```

```python
classifier = NaiveBayesRGBClassifier(colors)
classifier.accuracy()
```

    0.26806212018906145

That's not very good accuracy. Let's try different colorspaces:

```python
import colorsys
import sys

import husl
from colormath.color_conversions import convert_color
from colormath.color_objects import CMYKColor, LabColor, sRGBColor

# we'll run each classifier multiple times and look at the
# mean and standard deviation over all of the runs
RUNS_PER_CLASSIFIER = 5

class NaiveBayesHSVClassifier(NaiveBayesClassifier):
    def get_features(self, color):
        hsv = colorsys.rgb_to_hsv(*color.rgb)
        return dict(zip(("hue", "saturation", "value"), hsv))

class NaiveBayesHLSClassifier(NaiveBayesClassifier):
    def get_features(self, color):
        hls = colorsys.rgb_to_hls(*color.rgb)
        return dict(zip(("hue", "lightness", "saturation"), hls))

class NaiveBayesHUSLClassifier(NaiveBayesClassifier):
    def get_features(self, color):
        hsl = husl.rgb_to_husl(*color.rgb)
        return dict(zip(("hue", "saturation", "lightness"), hsl))

class NaiveBayesCMYKClassifier(NaiveBayesClassifier):
    def get_features(self, color):
        rgb = sRGBColor(*color.rgb)
        cmyk = convert_color(rgb, CMYKColor)
        return dict(zip(("cyan", "magenta", "yellow", "black"), (getattr(cmyk, v) for v in CMYKColor.VALUES)))

class NaiveBayesLabClassifier(NaiveBayesClassifier):
    def get_features(self, color):
        rgb = sRGBColor(*color.rgb)
        lab = convert_color(rgb, LabColor)
        return dict(zip(("lightness", "green-red", "blue-yellow"), (getattr(lab, v) for v in LabColor.VALUES)))
```

```python
results = [
    (c.__name__, c(colors).accuracy())
    for c in [
        NaiveBayesCMYKClassifier, NaiveBayesHLSClassifier, NaiveBayesHSVClassifier,
        NaiveBayesHUSLClassifier, NaiveBayesLabClassifier, NaiveBayesRGBClassifier]
    for _ in range(RUNS_PER_CLASSIFIER)
]
```


```python
pd.DataFrame(results, columns=['Classifier', 'Accuracy']) \
  .groupby('Classifier') \
  .agg({'Accuracy': ['mean', 'std']}) \
  .reset_index() \
  .sort_values(('Accuracy', 'mean'), ascending=False)
```

<div>
<table>
  <thead>
    <tr>
      <th></th>
      <th>Classifier</th>
      <th colspan="2" halign="left">Accuracy</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>mean</th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaiveBayesCMYKClassifier</td>
      <td>0.403241</td>
      <td>0.010765</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaiveBayesHSVClassifier</td>
      <td>0.370155</td>
      <td>0.010850</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaiveBayesHLSClassifier</td>
      <td>0.360702</td>
      <td>0.001464</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaiveBayesRGBClassifier</td>
      <td>0.262255</td>
      <td>0.007603</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaiveBayesLabClassifier</td>
      <td>0.226604</td>
      <td>0.011524</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaiveBayesHUSLClassifier</td>
      <td>0.225928</td>
      <td>0.010810</td>
    </tr>
  </tbody>
</table>
</div>

~40% still isn't great. Let's try a different kind of classifier.

### k-Nearest Neighbor

Let's try [k-NN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) classifiers over these colorspaces and values of N.

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class KNNColorClassifier(ColorFamilyClassifier):
    def __init__(self, color_corpus, train_percent=0.8, n_neighbors=5):
        # labels in the KNeighborsClassifier are integers, so we'll create
        # a unique integer label for each color family and map both ways for convenience
        families = color_corpus['family'].unique()
        self.family_map = {f: i for i, f in enumerate(families)}
        self.reverse_family_map = {v: k for k, v in self.family_map.items()}

        self.n_neighbors = n_neighbors
        super().__init__(color_corpus, train_percent)

    def get_label(self, color):
        return self.family_map[color.family]

    def init_classifier(self):
        self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        [features, labels] = zip(*self.train_set)
        self.classifier.fit(features, labels)

    def accuracy(self):
        [features, labels] = zip(*self.test_set)
        return self.classifier.score(features, labels)

    def classify(self, color):
        return self.reverse_family_map[
            self.classifier.predict(
                [self.get_features(color)]
            )[0]
        ]

    def get_neighbors(self, color):
        return self.shuffled.iloc[[
            i for i in self.classifier.kneighbors(
                np.array(self.get_features(color)).reshape(1, -1),
                return_distance=False
            )[0]
        ]]
```

```python
# we'll try values of n_neighbors in this range
MIN_N = 1
MAX_N = 20

class KNNRGBClassifier(KNNColorClassifier):
    def get_features(self, color):
        return color.rgb

class KNNHSVClassifier(KNNColorClassifier):
    def get_features(self, color):
        return colorsys.rgb_to_hsv(*color.rgb)

class KNNHLSClassifier(KNNColorClassifier):
    def get_features(self, color):
        return colorsys.rgb_to_hls(*color.rgb)

class KNNHUSLClassifier(KNNColorClassifier):
    def get_features(self, color):
        return husl.rgb_to_husl(*color.rgb)

class KNNCMYKClassifier(KNNColorClassifier):
    def get_features(self, color):
        rgb = sRGBColor(*color.rgb)
        cmyk = convert_color(rgb, CMYKColor)
        return tuple(getattr(cmyk, v) for v in CMYKColor.VALUES)

class KNNLabClassifier(KNNColorClassifier):
    def get_features(self, color):
        rgb = sRGBColor(*color.rgb)
        lab = convert_color(rgb, LabColor)
        return tuple(getattr(lab, v) for v in LabColor.VALUES)
```

```python
results = [
    (c.__name__, n, c(colors, n_neighbors=n).accuracy())
    for c in [
        KNNCMYKClassifier, KNNHLSClassifier, KNNHSVClassifier,
        KNNHUSLClassifier, KNNLabClassifier, KNNRGBClassifier]
    for n in range(MIN_N, MAX_N + 1)
    for _ in range(RUNS_PER_CLASSIFIER)
]
```

```python
pd.DataFrame(results, columns=['Classifier', 'N', 'Accuracy']) \
            .groupby(['Classifier', 'N']) \
            .agg({'Accuracy': ['mean', 'std']}) \
            .reset_index() \
            .sort_values(('Accuracy', 'mean'), ascending=False) \
            .head(5)
```

<div>
<table>
  <thead>
    <tr>
      <th></th>
      <th>Classifier</th>
      <th>N</th>
      <th colspan="2" halign="left">Accuracy</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th>mean</th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>95</th>
      <td>KNNLabClassifier</td>
      <td>16</td>
      <td>0.742606</td>
      <td>0.004173</td>
    </tr>
    <tr>
      <th>93</th>
      <td>KNNLabClassifier</td>
      <td>14</td>
      <td>0.737205</td>
      <td>0.006377</td>
    </tr>
    <tr>
      <th>99</th>
      <td>KNNLabClassifier</td>
      <td>20</td>
      <td>0.735044</td>
      <td>0.004930</td>
    </tr>
    <tr>
      <th>88</th>
      <td>KNNLabClassifier</td>
      <td>9</td>
      <td>0.734504</td>
      <td>0.006691</td>
    </tr>
    <tr>
      <th>96</th>
      <td>KNNLabClassifier</td>
      <td>17</td>
      <td>0.734099</td>
      <td>0.006909</td>
    </tr>
  </tbody>
</table>
</div>

It looks like the [Lab colorspace](https://en.wikipedia.org/wiki/Lab_color_space) was the most accurate, and 16 neighbors seems to have slightly outperformed other values in our range.

~74% accuracy should be "good enough" for our purposes.
Let's put this classifier to work!


```python
classifier = KNNLabClassifier(colors, train_percent=1.0, n_neighbors=16)
```

### Testing out the Classifier

Let's see how far off the classifier is when it's wrong. Is it pretty close (e.g. classifying an orange as a red or a yellow) or way off (e.g. classifying a blue as a pink)?

```python
classified = pd.DataFrame(
    ((c.family, classifier.classify(c)) for c in colors.itertuples()),
    columns=['Expected', 'Actual'])

totals = classified.groupby(['Expected']).size()
results = classified.groupby(['Expected', 'Actual']).size().reset_index(name='Count')
results['Pct'] = results.apply(lambda x: x['Count'] / totals.loc[x['Expected']], axis=1)
```

First, let's see which families the classifier most accurately identifies.


```python
results[results['Expected'] == results['Actual']] \
       .sort_values('Pct', ascending=False)
```

<div>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Expected</th>
      <th>Actual</th>
      <th>Count</th>
      <th>Pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32</th>
      <td>GREEN</td>
      <td>GREEN</td>
      <td>996</td>
      <td>0.869869</td>
    </tr>
    <tr>
      <th>69</th>
      <td>RED</td>
      <td>RED</td>
      <td>812</td>
      <td>0.865672</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BLUE</td>
      <td>BLUE</td>
      <td>836</td>
      <td>0.840201</td>
    </tr>
    <tr>
      <th>50</th>
      <td>ORANGE</td>
      <td>ORANGE</td>
      <td>680</td>
      <td>0.788863</td>
    </tr>
    <tr>
      <th>62</th>
      <td>PURPLE</td>
      <td>PURPLE</td>
      <td>504</td>
      <td>0.775385</td>
    </tr>
    <tr>
      <th>79</th>
      <td>WHITE</td>
      <td>WHITE</td>
      <td>262</td>
      <td>0.772861</td>
    </tr>
    <tr>
      <th>23</th>
      <td>GRAY</td>
      <td>GRAY</td>
      <td>448</td>
      <td>0.746667</td>
    </tr>
    <tr>
      <th>87</th>
      <td>YELLOW</td>
      <td>YELLOW</td>
      <td>556</td>
      <td>0.725849</td>
    </tr>
    <tr>
      <th>11</th>
      <td>BROWN</td>
      <td>BROWN</td>
      <td>538</td>
      <td>0.712583</td>
    </tr>
    <tr>
      <th>0</th>
      <td>BLACK</td>
      <td>BLACK</td>
      <td>35</td>
      <td>0.686275</td>
    </tr>
    <tr>
      <th>41</th>
      <td>NEUTRAL</td>
      <td>NEUTRAL</td>
      <td>75</td>
      <td>0.261324</td>
    </tr>
  </tbody>
</table>
</div>

Next, we'll take a look at each family and see which family it is most commonly incorrectly identified as.

```python
results[results['Expected'] != results['Actual']] \
       .sort_values('Pct', ascending=False) \
       .drop_duplicates('Expected')
```

<div>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Expected</th>
      <th>Actual</th>
      <th>Count</th>
      <th>Pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>56</th>
      <td>PINK</td>
      <td>RED</td>
      <td>14</td>
      <td>0.823529</td>
    </tr>
    <tr>
      <th>39</th>
      <td>NEUTRAL</td>
      <td>GRAY</td>
      <td>89</td>
      <td>0.310105</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BLACK</td>
      <td>GRAY</td>
      <td>9</td>
      <td>0.176471</td>
    </tr>
    <tr>
      <th>81</th>
      <td>YELLOW</td>
      <td>BROWN</td>
      <td>93</td>
      <td>0.121410</td>
    </tr>
    <tr>
      <th>47</th>
      <td>ORANGE</td>
      <td>BROWN</td>
      <td>66</td>
      <td>0.076566</td>
    </tr>
    <tr>
      <th>19</th>
      <td>BROWN</td>
      <td>YELLOW</td>
      <td>49</td>
      <td>0.064901</td>
    </tr>
    <tr>
      <th>60</th>
      <td>PURPLE</td>
      <td>GRAY</td>
      <td>42</td>
      <td>0.064615</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BLUE</td>
      <td>GREEN</td>
      <td>60</td>
      <td>0.060302</td>
    </tr>
    <tr>
      <th>65</th>
      <td>RED</td>
      <td>BROWN</td>
      <td>55</td>
      <td>0.058635</td>
    </tr>
    <tr>
      <th>21</th>
      <td>GRAY</td>
      <td>BLUE</td>
      <td>35</td>
      <td>0.058333</td>
    </tr>
    <tr>
      <th>75</th>
      <td>WHITE</td>
      <td>NEUTRAL</td>
      <td>15</td>
      <td>0.044248</td>
    </tr>
    <tr>
      <th>35</th>
      <td>GREEN</td>
      <td>YELLOW</td>
      <td>44</td>
      <td>0.038428</td>
    </tr>
  </tbody>
</table>
</div>

This makes sense as we have very few datapoints in the pink family, neutral and gray have a lot of overlap, and most of the paint colors in the black family are actually gray.

For now let's just accept this as "good enough" and have some fun. Let's generate a some random RGB values and take a guess at the family to which it belongs.


```python
import random

def random_colors(i=1):
    return pd.DataFrame({
        'name': '???',
        'family': '???',
        'rgb': (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
    } for _ in range(i))

def classify_colors(df):
    return df.apply(classifier.classify, axis=1)
```

```python
new_colors = random_colors(5)
new_colors['family'] = classify_colors(new_colors)
display_colors(new_colors)
```

<div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">???</div>GREEN</p>
      <p><svg width="64" height="64" style="background: #228c42" /></p>
      <p>#228c42</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">???</div>BLUE</p>
      <p><svg width="64" height="64" style="background: #42d5f0" /></p>
      <p>#42d5f0</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">???</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #826f9f" /></p>
      <p>#826f9f</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">???</div>GREEN</p>
      <p><svg width="64" height="64" style="background: #64e599" /></p>
      <p>#64e599</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">???</div>BROWN</p>
      <p><svg width="64" height="64" style="background: #4d360b" /></p>
      <p>#4d360b</p>
  </div>
</div>

# 'Desert Rose' by Any Other Name

Now that we can make a decent guess towards the color family for a random RGB value, let's try to build off of the existing color names for similar colors to create fun new names.

Let's start by creating our random mystery color.

```python
mystery_color = new_colors.sample(1)
mystery_color['family'] = classify_colors(mystery_color)
display_colors(mystery_color)
```

<div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">???</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #826f9f" /></p>
      <p>#826f9f</p>
  </div>
</div>

Let's look at the closest named colors.

```python
neighbors = classifier.get_neighbors(mystery_color.iloc[0])
display_colors(neighbors)
```

<div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">CLEMATIS</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #7e6596" /></p>
      <p>#7e6596</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">FORGET-ME-NOT</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #716998" /></p>
      <p>#716998</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">PURPLE AGATE</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #8c7eaf" /></p>
      <p>#8c7eaf</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">PURPLE PARADISE</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #79669e" /></p>
      <p>#79669e</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">CHARMED VIOLET</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #8b7eb1" /></p>
      <p>#8b7eb1</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">ROMANTIC MOMENT</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #8f76af" /></p>
      <p>#8f76af</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">SECOND POUR</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #887ca5" /></p>
      <p>#887ca5</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">NOTORIOUS</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #7b658b" /></p>
      <p>#7b658b</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">LILAC INTUITION</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #997ea8" /></p>
      <p>#997ea8</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">VIOLET VIXEN</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #74688c" /></p>
      <p>#74688c</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">UNIMAGINABLE</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #8b7eba" /></p>
      <p>#8b7eba</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">SEDUCTION</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #655f8e" /></p>
      <p>#655f8e</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">NAPLES SUNSET</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #987ea4" /></p>
      <p>#987ea4</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">CROCUS PETAL PURPLE</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #9487ba" /></p>
      <p>#9487ba</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">KIMONO VIOLET</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #75769c" /></p>
      <p>#75769c</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">CLASSIC WALTZ</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #71588d" /></p>
      <p>#71588d</p>
  </div>
</div>

## On your Markov, get set, go!

To generate names for our mystery color, let's try training a [Markov chain](https://en.wikipedia.org/wiki/Markov_chain) on not only the names of these closest colors, but the product of all synonyms of all component words within the names to give us more variety. We'll limit synonyms by part of speech so the generated names make slightly more sense.

```python
from nltk.corpus import wordnet
import spacy # faster than wordnet for tokenizing and part-of-speech tagging

nlp = spacy.load("en")

# map spaCy POS to WordNet
POS_MAP = {
    'ADJ': 'a',
    'ADV': 'r',
    'NOUN': 'n',
    'VERB': 'v',
}

def get_syns(token):
    """get synonyms for a spaCy token"""
    synsets = wordnet.synsets(token.orth_, pos=POS_MAP.get(token.pos_))
    if synsets:
        return itertools.chain.from_iterable(s.lemma_names() for s in synsets)
    return [token.orth_]

def explode(color_name):
    """explode a color name into the product of all of its component words' synonyms"""
    return set(
        ' '.join(variant).replace('_', ' ').upper()
        for variant in itertools.product(
            *(get_syns(token) for token in nlp(color_name.lower()))))
```

```python
import string

import markovify

def make_markov_model(colors):
    return markovify.Text(None, # we're pre-parsing the sentences
        parsed_sentences=[
            variant.split()
            for variant in set(
                itertools.chain.from_iterable(
                    colors['name'].apply(explode).values))
        ])

def name_color(color):
    model = make_markov_model(classifier.get_neighbors(color))
    return string.capwords(
        model.make_sentence(
            # we're generating short names and don't care about overlap with original text
            test_output=False, max_words=3))
```

```python
mystery_color['name'] = name_color(mystery_color.iloc[0])
display_colors(mystery_color)
```

<div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">Over-embellished Eden</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #826f9f" /></p>
      <p>#826f9f</p>
  </div>
</div>

Not bad for a computer. Let's try it some more!

```python
new_colors['name'] = new_colors.apply(name_color, axis=1)
display_colors(new_colors)
```

<div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">Unmediated Dark-green</div>GREEN</p>
      <p><svg width="64" height="64" style="background: #228c42" /></p>
      <p>#228c42</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">Queen Story Drab</div>BLUE</p>
      <p><svg width="64" height="64" style="background: #42d5f0" /></p>
      <p>#42d5f0</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">Lilac Suspicion</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #826f9f" /></p>
      <p>#826f9f</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">Branch Brook Dark-green</div>GREEN</p>
      <p><svg width="64" height="64" style="background: #64e599" /></p>
      <p>#64e599</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">Crisp Browned</div>BROWN</p>
      <p><svg width="64" height="64" style="background: #4d360b" /></p>
      <p>#4d360b</p>
  </div>
</div>

Most of the generated names will be nonsensical (and many also NSFW), but I did come across a few good ones. Here are the highlights:

<div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">Unprompted Empurpled</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #6e3281" /></p>
      <p>#6e3281</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">Million Dollar Marxist</div>RED</p>
      <p><svg width="64" height="64" style="background: #b70a1f" /></p>
      <p>#b70a1f</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">Induce Watercourse</div>BLUE</p>
      <p><svg width="64" height="64" style="background: #10cffb" /></p>
      <p>#10cffb</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">Sharp-worded American Cheddar</div>ORANGE</p>
      <p><svg width="64" height="64" style="background: #e77616" /></p>
      <p>#e77616</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">Summertime Sorry</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #8fb0eb" /></p>
      <p>#8fb0eb</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">Graeco-roman Chocolate-brown</div>BROWN</p>
      <p><svg width="64" height="64" style="background: #52382e" /></p>
      <p>#52382e</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">Cat Valium</div>GREEN</p>
      <p><svg width="64" height="64" style="background: #5bfee2" /></p>
      <p>#5bfee2</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">Unconscionable Orange Tree</div>ORANGE</p>
      <p><svg width="64" height="64" style="background: #c44312" /></p>
      <p>#c44312</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">Unused Butter</div>YELLOW</p>
      <p><svg width="64" height="64" style="background: #f0d299" /></p>
      <p>#f0d299</p>
  </div>
  <div style="width: 148px; display: inline-block">
      <p><div style="font-weight: bold">Scandalmongering Shuttlecock</div>YELLOW</p>
      <p><svg width="64" height="64" style="background: #e8c467" /></p>
      <p>#e8c467</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">Norse Naughty</div>PURPLE</p>
      <p><svg width="64" height="64" style="background: #5d04f2" /></p>
      <p>#5d04f2</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">Disconsolate Denim</div>BLUE</p>
      <p><svg width="64" height="64" style="background: #74b7c5" /></p>
      <p>#74b7c5</p>
  </div>
  <div style="width: 148px; display: inline-block">
      <p><div style="font-weight: bold">Italian Methamphetamine Green</div>GREEN</p>
      <p><svg width="64" height="64" style="background: #84e8e9" /></p>
      <p>#84e8e9</p>
  </div>
  <div style="width: 128px; display: inline-block">
      <p><div style="font-weight: bold">Odoriferous & Off-key</div>YELLOW</p>
      <p><svg width="64" height="64" style="background: #d7b743" /></p>
      <p>#d7b743</p>
  </div>
</div>

## Journey's End (#BAC9D6)

For now it's time to climb back out of the rabbit hole, but maybe one day we can [teach our algorithm about puns](https://arxiv.org/pdf/1704.08224.pdf) (or even just include homophones in addition to synonyms to increase the likelihood of accidental puns).

Thanks for humoring me and go have some [fun with computers]({{ site.url }}{{ site.baseurl }}/).

