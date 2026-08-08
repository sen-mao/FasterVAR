# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import os
import random
import time
from threading import Lock
from abc import abstractmethod
from typing import Any, List, Union, Optional

import numpy as np

import openai

# from auto_caption.models.base_model import Model
OPEN_API_KEY = os.environ.get('OPEN_API_KEY')
GLOBAL_AZURE_ENDPOINT = os.environ.get('GLOBAL_AZURE_ENDPOINT')

class SingletonArgMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """

    _instances = {}

    _lock: Lock = Lock()
    """
    We now have a lock object that will be used to synchronize threads during
    first access to the Singleton.
    """

    def __call__(cls, *args, **kwargs):
        """
        changes to the value of the `__init__` argument do affect
        the returned instance.
        """
        # Now, imagine that the program has just been launched. Since there's no
        # Singleton instance yet, multiple threads can simultaneously pass the
        # previous conditional and reach this point almost at the same time. The
        # first of them will acquire lock and will proceed further, while the
        # rest will wait here.
        with cls._lock:
            # The first thread to acquire the lock, reaches this conditional,
            # goes inside and creates the Singleton instance. Once it leaves the
            # lock block, a thread that might have been waiting for the lock
            # release may then enter this section. But since the Singleton field with
            # specific arguments is already initialized, the thread won't create a new object.
            if cls.__name__+str(args)+str(kwargs) not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls.__name__+str(args)+str(kwargs)] = instance
        return cls._instances[cls.__name__+str(args)+str(kwargs)]


class Model(metaclass=SingletonArgMeta):
    """an abstrct model"""

    def __init__(self, model_name: Union[str, List[str]], ak: Union[str, List[str]], token_stat_percent: Optional[float] = None) -> None:
        self.clients = self._init_clients(model_name, ak)
        if token_stat_percent is not None:
            self._init_token_stat(token_stat_percent)

    def _init_token_stat(self, token_stat_percent):
        self.token_stat_percent = token_stat_percent
        self.token_sort = []
        self.token_stat = {'max_token': 0, 'mean_token': 0,
                           'count': 0, f'p{token_stat_percent*100}_token_num': 0}
        self.token_stat_percent = token_stat_percent

    def _init_clients(self, model_name, ak):
        if not isinstance(model_name, list):
            model_name = [model_name]
        if not isinstance(ak, list):
            ak = [ak]
        clients = []
        if len(ak) > 1 and len(model_name) == 1:
            model_name = model_name*len(ak)
        elif len(ak) == 1 and len(model_name) > 1:
            ak = ak*len(model_name)

        assert len(ak) == len(
            model_name), f"length of ak = {len(ak)} != length of model_name = {len(model_name)}"
        for model, ak in zip(model_name, ak):
            client = self._creat_client(model, ak)
            clients.append(client)
        print(f"init {len(clients)} clients!!!")
        return clients

    def _update(self, token_num):
        self.token_sort.append(token_num)
        self.token_stat[f'p{self.token_stat_percent*100}_token_num'] = round(
            np.percentile(self.token_sort, self.token_stat_percent*100), 2)
        self.token_stat['count'] = len(self.token_sort)
        self.token_stat['mean_token'] = round(np.mean(self.token_sort), 2)
        self.token_stat['max_token'] = np.max(self.token_sort)

    @abstractmethod
    def _creat_client(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError


class OpenAIGPTModel(Model):
    _global_azure_endpoint = GLOBAL_AZURE_ENDPOINT 
    _api_version = "2023-05-15"

    def __init__(self, model_name='gpt-4', ak='', log_prob=0.01, if_global=False, token_stat_percent=0.99) -> None:
        self.ak = OPEN_API_KEY if not ak else ak
        self.if_global = if_global
        self.ak_state = {}
        self.ak_state_succ = {}
        self.log_prob = log_prob
        self.start_time = time.time()
        super().__init__(model_name, ak, token_stat_percent)

    def _creat_client(self, model_name, ak):
        client = openai.AzureOpenAI(
            azure_endpoint=OpenAIGPTModel._global_azure_endpoint,
            api_version=OpenAIGPTModel._api_version,
            api_key=ak,
        )
        client.temp_model_name = model_name
        client.temp_ak = ak
        self.ak_state[ak[:5]] = 0
        self.ak_state_succ[ak[:5]] = 0
        return client

    def __call__(self, prompt="hello", system_prompt=None, max_tokens=1000, return_output_token_length=False):
        client = random.choice(self.clients)
        ak = client.temp_ak
        self.ak_state[ak[:5]] += 1
        if random.random() < self.log_prob:
            for ak in self.ak_state:
                print(
                    f"ak: {ak} 请求数：{self.ak_state[ak]}, 成功数：{self.ak_state_succ[ak]}, 成功率：{self.ak_state_succ[ak]/self.ak_state[ak]*100:.2f}%, 速度：{self.ak_state_succ[ak]/(time.time()-self.start_time)*60:.2f} 个/分钟")
            print(f"token_stat：{self.token_stat}")

        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        completion = client.chat.completions.create(
            extra_headers={"X-TT-LOGID": "lizhe.xyz"},  # 请务必带上此header，方便定位问题
            model=client.temp_model_name,
            messages=messages,
            max_tokens=max_tokens
        )
        self.ak_state_succ[ak[:5]] += 1
        if self.token_stat_percent is not None:
            # 输出token统计
            self._update(completion.usage.completion_tokens)
        
        if return_output_token_length:
            # 允许输出token长度，便于筛选因输出达到max_tokens而被截断的数据
            return completion.choices[0].message.content, completion.usage.completion_tokens
        return completion.choices[0].message.content


# better prompt
system_prompt = '''You are a prompt engineer, aiming to rewrite user inputs into high-quality prompts for better video generation without affecting the original meaning.\n''' \
'''Task requirements:\n''' \
'''1. If no subject appearance description provided in user inputs, add some descriptions about appearance (e.g., clothing, gender);\n''' \
'''2. If there is no camera movement information in the user inputs and the entire scene is very static, you can appropriately add some simple camera movement descriptions;\n''' \
'''3. If there is no description of the setting, add some visual details about the scene, but there should not be a change of the scene;\n''' \
'''4. If there is no description of shot scale in the user's input, you can appropriately add some information about the shot scale;\n''' \
'''5. If the character's actions take too little time, you can appropriately add some detailed actions during or after the original action so that the whole prompt described can last for 5 seconds in total. The newly added actions need to be dynamic rather than static (for example, looking into the distance or standing still). Add the main subject's actions in chronological order, and include detailed descriptions of their physical movements;\n''' \
'''6. If you are describing an action, describe the entire process, for example, "pick up the clothes, put your hands into the sleeves, and put on the clothes.";\n''' \
'''7. The word count for each action description should be evenly distributed throughout the entire output, and there should not be any action descriptions that are significantly longer or shorter than the others;\n''' \
'''8. The entire sequence of actions needs to be engaging and impressive;\n''' \
'''9. If there are multiple subjects, describe the actions of each subject;\n''' \
'''10. All the descriptions must be about appearance or action, and no sensory, feeling or highlighting descriptions are allowed;\n''' \
'''11. Do not include any emphatic or highlighted descriptions such as demonstrating, showcasing, highlighting, emphasizing anything in the rewritten text;\n''' \
'''12. All descriptions should be as objective as possible;\n''' \
'''13. Unless the subject refers to multiple objects, use he/she/it instead of they to refer to the subject;\n''' \
'''14. In describing actions, do not add adverbs before verbs, do not add prepositional phrase (e.g., with something) after verbs.\n''' \
'''15. Output the entire prompt in English, retaining original text in quotes and titles, and preserve key input information;\n''' \
'''16. If the original input is purely a landscape description, do not add additional subjects such as people;\n''' \
'''17. Who does what action? Clarify the subject at the beginning of the sentence;\n''' \
'''18. If you describe using the actions "Rolling", "raising hands", "lifting legs", etc., clarity whether they are forward or backward;\n''' \
'''19. If the prompts specify the direction of movement, do not adjust it at will;\n''' \
'''20. If describing a situation with multiple people, such as a group, state the number of people, but not too many, usually less than five;\n''' \
'''21. If you are describing a person rotating, if you are describing the head turning, you should also describe the body turning as well;\n''' \
'''22. When you describe actions, background and camera movement, you need to write them separately, don't mix them together;\n''' \
'''23. If the original input is related to anime, you need to add "This video showcases an animated scene" at the beginning of your rewritten prompt. If there is a requirement for 2D or 3D animation, you should also include that information;\n''' \
'''24. The revised prompt should be between 50-140 words long and always use simple and direct words.\n''' \

'''Original user inputs and the examples after rewriting:\n''' \

'''1. 
- **Original user inputs**: A woman is swimming underwater in a pool, extending her arms forward and kicking her legs. The pool has colorful lane dividers in the background. She performs a somersault, rotating her body in the water. After the somersault, she continues swimming forward with a streamlined body position.
- **Examples after rewriting**: The video captures a female swimmer underwater, showing a sequence of movements that demonstrate various swimming techniques. Initially, the swimmer's body is horizontal, with the arms extended forward and the legs straightened, indicating the start of a stroke cycle. As the video progresses, the head turns slightly to the side, the arms begin to bend, and the legs begin to move in a rhythmic dolphin kick, creating a powerful propulsion mechanism. Throughout the frame, the swimmer maintains an streamlined position with minimal movement of the hands and feet to maintain balance and continue forward momentum. The swimmer is wearing a white bathing suit with pink floral patterns, her hair tied back to avoid obstructions. Bubbles can be seen around her body, indicating that she is moving through the water with fluid movements. In the background is a swimming pool lane marked by red and yellow lane lines that extend across the pool. The clear blue water reflects the sunlight, creating a serene and vibrant atmosphere. Another person is visible in the distance, partially obscured by the water's surface, swimming in the same lane. The swimmer's movements create small splashes and bubbles as she moves through the water. The camera follows her in front, highlighting the technique and form of the swimmer. The overall scene conveys a sense of focus and athleticism, highlighting the skill and grace of the swimmer in the water. The camera moves with the female athlete, keeping her in the center of the frame.\n''' \

'''2. 
- **Original user inputs**: In an indoor tennis court, a man prepares to serve a tennis ball. He tosses the ball into the air and swings his racket to hit it. The opponent moves to the right to intercept the ball. The ball hits the net and falls to the ground on the opponent's side. The man in the white shirt follows through with his serve and then moves to the left side of the court.
- **Examples after rewriting**: The video captures a well-lit indoor tennis court with high ceilings and large windows that allow natural light to filter in. The court has a standard blue surface with white borders and a net in the middle. A male player in a white T-shirt and black shorts can be seen tossing a ball upward with his left hand and swinging his racket at the ball with his right hand. The opposing player in a white top and dark shorts then moves to the right of the frame, opens his racket, runs after the ball, and returns a powerful forehand shot that hits the net and lands. The background includes an advertisement for tennis equipment brand Babolat and a bench with practice tennis balls. The overall atmosphere reflects focused training in a professional environment. The camera focuses on the man, moving first to the left and then to the right.\n''' \

'''3. 
- **Original user inputs**: A man wearing black athletic clothing and bright orange running shoes is running on a paved track. The background features a large, green grassy field with scattered trees and a few buildings in the distance. The sky is clear and blue, indicating a sunny day. The man maintains a steady pace throughout the sequence, with his arms bent at the elbows and his legs moving in a rhythmic motion.
- **Examples after rewriting**: The video captures a male runner in motion in an outdoor setting. The man is shirtless, wearing black running pants and bright orange running shoes. He has a black headband on his head and his hair is closely cropped. He runs on the road, swinging his hands back and forth at his sides as he runs, and his left and right legs alternate between landing and lifting on the road. The man maintains a consistent posture and runs at a constant speed, showing a continuous running posture. The background is a large green lawn that looks like a park or sports field, with trees scattered around the edges. In the distance, buildings that look like houses or apartments can be seen, indicating that the location is close to a residential area. The sky is clear and blue, indicating that it is a sunny day with good weather. The camera follows the man as he moves to the left.\n''' \

'''4. 
- **Original user inputs**: In a large indoor arena with many people and banners, two individuals dressed in fencing gear are engaged in a fencing match. They are positioned on a white mat with a blue mat nearby. The fencers are seen lunging and parrying at each other with their swords. After a series of movements, they begin to separate and walk towards the right side of the frame. The fencer on the right raises their arm and points towards a man dressed in black standing near a table with a 'naked' banner.
- **Examples after rewriting**: The video captures an intense fencing match between two male fencers in a large indoor arena. The fencers are dressed in traditional white fencing gear, including masks, jackets, gloves, and trousers, with one fencer wearing a light-colored top and light-colored pants, a yellow and black socks and shoes, and a mask with a blue logo on it, while the other fencer has a mask with a red logo on it. They are both holding fencing foils in their hands and positioned on a white mat with a blue mat nearby, with bags and equipment scattered nearby. The male fencer on the left side of the frame, wearing a mask with a blue logo on it, leans back with his legs bent, his left hand extended forward holding the sword, and then he stands up straight and continues to attack, while the male fencer on the right side of the frame, wearing a mask with a red logo on it, stands there with his left foot in front and his right foot behind, his right hand extended forward holding the sword, and then both of them stand up straight and continue to attack and defend. The fencer on the right side of the frame actively moves forward with his right foot in front and his left foot behind, while the fencer on the left side of the frame extends his left leg and swings his left hand to attack, while the fencer on the right side of the frame swings his right hand to defend. The fencer on the left side of the frame then retracts his left hand and turns to leave, while the fencer on the right side of the frame turns around and walks away while raising his right hand and looking at the referee dressed in black near the table with the "Naked" banner. The background shows a busy arena with spectators sitting in the stands and other fencers or staff near the barriers. The banners on the fence read "USA Fencing" and "Naked", indicating that this is part of an official event. The high ceiling is supported by structural pillars, and bright lights illuminate the entire space. The atmosphere is focused and competitive, which is typical of competitive fencing events. The camera lens follows the two fencers as they move.\n''' \

'''5. 
- **Original user inputs**: Players in red and white uniforms are actively engaged in the game, with the player in white number 5 dribbling the ball. The player in white number 5 continues to dribble and moves towards the basket, closely followed by players in red uniforms. The player in white number 5 attempts a shot while being closely guarded by the players in red. The player in white number 5 successfully makes a basket, and the players in red attempt to block the shot. The ball goes through the hoop, scoring a point. The players in red and white continue to move around the court, with the player in white number 5 preparing to make another move.
- **Examples after rewriting**: The video records a lively men's basketball game, which is played in an indoor gymnasium. The gymnasium has high ceilings and exposed metal beams. The court is marked with regulation lines, and the walls are painted white and decorated with colorful banners and a large red banner with yellow Chinese characters. A group of 10 players in red and white jerseys are actively engaged in the game. A player wearing a white No. 5 jersey is moving towards the basket with the ball, while his opponent in a red jersey is in hot pursuit. As the play progresses, the player with the ball fakes first to the right and then to the left, bypassing the defender and shooting towards the basket. The ball arcs through the air and enters the basket, while other players move to grab rebounds or assists. Another player in a red jersey jumps up to try to block the shot. Players on the sidelines watch the game intently, with referees or coaches nearby, with some of whom sit at tables holding referee's referee documents. The bright gym lights ensure that the game is clearly visible, highlighting the competitive nature of the sport and the strategic interactions between players. The camera follows the movement of player No. 5.\n''' \

'''6. 
- **Original user inputs**: In an indoor badminton court with green walls and a balcony, two men are engaged in a badminton game. The man in black hits the shuttlecock, and the man in white returns it.
- **Examples after rewriting**: The video captures a dynamic game of badminton in an indoor court with a green floor and white boundary lines. The court is surrounded by green walls and has wooden floors and fixed badminton nets. There is a blue banner with white Chinese characters on the left wall and a brand logo on the right. Initially, the male player near the camera, wearing a black shirt and shorts, holds the racket in his right hand behind him and prepares to hit the ball, while the male player wearing a white shirt and dark shorts stands in a ready position on the opposite court. Both players are focused on the game, moving quickly and strategically. The black player hits the shuttlecock with the racket in his right hand extended forward with force, and the white player hits the shuttlecock with the racket in his right hand extended forward with force. Following the same pattern, the intensity of the game keeps them fully engaged, as they move across the court to return the shuttlecock and maintain their positions. Towards the end of the video, the shuttlecock appears to have been hit out of the court, indicated by the players' continued movements and gestures. In the background, there are several stationary exercise bikes against the wall and a few other people sitting or standing, perhaps observing or resting. The bright lights in the room highlight the action, emphasizing the speed and precision required for each shot. The video ends with the black player running to the left of the frame and the white player preparing to hit the shuttlecock with the racket in his right hand. The camera perspective does not change. The overall atmosphere reflects a competitive yet friendly sporting environment.\n''' \

'''7. 
- **Original user inputs**: A man stands on a stone platform by the river. The man releases his grip and tucks his body into a forward roll. The man dives into the river, creating a splash upon entry.
- **Examples after rewriting**: The video captures a man diving into a stone pool from a stone platform. The man is shirtless, wearing black shorts with white stripes and a black hat. Initially, the man stands on one leg on the edge of the platform, stretching his arms upward, ready to stretch. He then lowers his arms and prepares to jump, swinging his arms back and forth as he begins to move. The man leaps forward, leaning forward, stretching his arms forward, and his feet alternately downward, completing a somersault before entering the water, with his arms extended upward at the top and your legs extended upward at the bottom. As he enters the water, he creates a splash. The other man, also shirtless and wearing black shorts, stands nearby, observing the diver. The pool is surrounded by stone walls and has a traditional architectural style, with trees and buildings visible in the background. The calm water reflects the sunlight and ripples, highlighting the contrast between stillness and the splash created by the diver's entry. The camera remains stationary, capturing the action as it unfolds against this serene yet dynamic backdrop.\n''' \

'''8. 
- **Original user inputs**: In an indoor table tennis facility with multiple tables, a man and a woman are engaged in a game of table tennis.
- **Examples after rewriting**: The video captures an intense game of table tennis in an indoor gymnasium. The venue is spacious, with several tables and chairs neatly arranged in the background. The floor is covered with a red non-woven fabric, and the walls are painted beige and white, with overhead lights evenly illuminating the field. In the video, a man in a yellow shirt and black pants is playing against a woman wearing a white jacket, beige pants and white sneakers. At first, the man in yellow stands on the left side of the screen, leaning forward, holding the racket in right hand and stretching his arms forward, while the woman in white stands on the right side of the screen, ready to receive the ball. The man in yellow serves, and the woman in white receives the ball. Then the man in yellow and the woman in white both stand up straight and prepare to hit back. The two exchange blows successfully, and the game becomes fierce. The man in yellow hits a powerful ball, but the woman in white didn't miss it, and then she hits a powerful ball. The man in yellow hits the ball again, and the woman in white jumps up and hits a powerful ball, making the man in yellow raise his arm and give up. The fierce competitive atmosphere reflects in their focused expressions and agile movements, highlighting the dynamic nature of this sport. The camera angle remains fixed, providing a clear perspective of the action as it unfolds.\n''' \

'''9. 
- **Original user inputs**: A woman stands on a blue gymnastics mat. She performs a series of spins and jumps, maintaining her balance and poise.
- **Examples after rewriting**: The video captures a female gymnast performing on a blue mat during a competition. The gymnast wears a black leotard with flowing pink patterns, which contrasts sharply against the blue mat. Her hair is tied into two small pigtails, showing her concentration and focus. First, she stretches her arms horizontally and stands in a squatting position, showing her flexibility and control. Then she turns her body while raising one arm and bending the other, showing her agility and grace. She leaps into the air, maintaining a dynamic pose with one leg extended forward and the other backward, before returning to a squatting position and completing a roll, rolling into a front flip in the air, spinning twice to land steadily. She then stands up straight and stretches her arms horizontally to complete her performance with a confident hand gesture and a smile. The venue is an indoor gymnasium, and spectators can be seen sitting in the stands in the background. Judges sit at a table with a banner that reads "OLAY", observing and evaluating the performance. The gymnast runs energetically across the mat, leaps into the air with precision, and transitions into complex movements with great skill. The audience remains focused throughout, highlighting the competitive atmosphere of the event. The camera follows her movements, capturing her in action.''' \

