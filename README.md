# MALNIS at IberLEF-2022 DETESTS Task: A Multi-Task Learning Approach for Low-Resource Detection of Racial Stereotypes in Spanish

Author | Personal Website | Email
---|---|---
Juan Ramirez-Orta | [Homepage](https://web.cs.dal.ca/~juanr/) | juan.ramirez.orta@dal.ca
María Virginia Sabando | | virginia.sabando@cs.uns.edu.ar
Mariano Maisonnave | [Homepage](https://web.cs.dal.ca/~maisonnave/) | mariano.maisonnave@dal.ca
Evangelos Milios | [Homepage](https://web.cs.dal.ca/~eem/) | eem@cs.dal.ca

## Abstract

This paper describes our submission for the DETESTS (DETEction and classification of racial STereo-types in Spanish) shared task at IberLEF 2022. The DETESTS shared task is divided into two sub-tasks: in the first one, the objective consists of detecting racial biases in online comments as a binary classification problem, whereas in the second one, the goal is to determine whether the comments exhibit one or more of ten different racial biases as a multi-label classification problem. Our approach consists of a Multi-Task Learning strategy applied to pre-trained deep language models, which allows to learn a sequence representation for each comment. This representation is then used to train a joint classifier for all the categories of the second task, combining them using 𝐿𝑂𝐺𝐼𝐶𝐴𝐿_𝑂𝑅 to produce the predictions for the first one. The intuition behind our approach is that the joint training process allows the model to leverage the information present in each one of the categories and benefit from how they complement each other, boosting the performance of those categories with less examples. Our approach obtained ninth place in the first task and first place in the second one.

* [Full Paper](http://ceur-ws.org/Vol-3202/detests-paper2.pdf)
* [10-minute Presentation](https://youtu.be/CTDgEEzjY1k)

## Features

## Installation

    git clone https://github.com/jarobyte91/detests_2022.git
    cd detests_2022
    pip install -r requirements.txt
    
## Contribute & Support 

* [Issue Tracker](https://github.com/jarobyte91/detests_2022/issues)
* [Pull Requests](https://github.com/jarobyte91/detests_2022/pulls)

## License

This project is licensed under the MIT License.
