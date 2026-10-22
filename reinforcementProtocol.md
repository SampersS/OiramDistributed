- environment approach
environments zijn een standaard voor games van open ai die de entities in de game doorgeven als coordinaten en ook een lijst van acties geeft. wij gebruiken (https://github.com/Kautenja/gym-super-mario-bros). Dit is nodig omdat de pies niet krachtig genoeg zijn om zelf object detection te doen op de game screencapture.

### gemini the gpt answer:

1. Hoe laat je een learning library communiceren met de environment?Dit gebeurt via de Gymnasium-standaard (API). Vrijwel alle moderne reinforcement learning-bibliotheken (zoals Stable-Baselines3) zijn zo ontworpen dat ze precies weten hoe ze met een env-object moeten praten.De bibliotheek roept automatisch env.reset() aan om het spel te starten.Het model berekent een actie en stuurt die naar de environment via env.step(action).De environment geeft de nieuwe state, de reward en of het spel klaar is (terminated) terug aan de bibliotheek.In code ziet dat er heel simpel uit: Je geeft de environment gewoon mee aan je model, bijvoorbeeld:Pythonfrom stable_baselines3 import PPO

model = PPO("CnnPolicy", env, verbose=1)  # 'env' wordt hier gekoppeld aan het PPO-algoritme
model.learn(total_timesteps=10000)

---

### 2. Waar steek je jouw eigen reward-functie?
Als je een eigen reward-functie wilt gebruiken die afwijkt van de standaardinstelling, gebruik je een **Gym Wrapper** (een schil om je environment heen). 

* **Hoe het werkt:** Je maakt een eigen Python-klasse die overerft van `gym.Wrapper`. 
* Binnen deze wrapper intercept je de `step()` functie. Wanneer Mario een stap zet, kijkt jouw wrapper naar de `info`-data (bijv. de exacte X-coördinaat of levens uit het RAM-geheugen). 
* Vervolgens bereken je je **eigen reward**, overschrijf je de standaard reward, en stuur je die aangepaste waarde door naar je leer-algoritme.

Voorbeeld van een simpele Wrapper-structuur:
```python
import gymnasium as gym


class CustomRewardWrapper(gym.Wrapper):

  def step(self, action):
    state, reward, terminated, truncated, info = self.env.step(action)

    # Schrijf hier jouw eigen reward-logica op basis van 'info'
    if info['x'] > 500:
      custom_reward = 100  # Extra beloning als Mario ver is
    else:
      custom_reward = reward  # Anders de standaard reward

    return state, custom_reward, terminated, truncated, info
```
3. Moet je zelf de NES ROM uploaden?Bij de bibliotheek gym-super-mario-bros hoeft dit gelukkig niet.Deze bibliotheek is zo ontworpen dat de benodigde NES-ROM automatisch wordt meegeleverd en geladen via de onderliggende emulator-koppeling (nes-py) wanneer je pip install gym-super-mario-bros draait.   Je hoeft dus zelf geen illegale ROM-bestanden van het internet te plukken of handmatig te uploaden; het werkt direct zodra je de package installeert. (Dit in tegenstelling tot bijvoorbeeld OpenAI Retro, waar je soms wel zelf ROMs moet importeren).
