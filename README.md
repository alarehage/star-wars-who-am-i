# An image classification app to identify Star Wars Characters

## Intro
Deployed at https://star-wars-who-am-i.herokuapp.com/.

For sound, use non-Chrome browsers as autoplay function is disabled on Chrome.

For a full list of identifiable characters, scroll to the bottom.

Usage:
1. Select an image
2. Press "Submit"
3. Get your prediction

## Model building and performance
Approach:
1. Crawled for images for each class
2. Filtered out wrong/bad images
3. Experimented with MobileNetv2 and ResNet50 for model training

Results:
- Weighted F1: 0.70
- Weighted Precision: 0.72
- Weighted Recall: 0.71
- Acc: 0.71

Findings:
- First data pull: 
    - MobileNetv2 produced an F1 of 0.57
    - Data issues: insufficient + not cleaned, some classes not clearly defined
- Second pull: 
    - ResNet50 achieved a max F1 of 0.64
    - Changed some class names to be better defined for pulling. Also added classes, totalling 67 (previously 42)
    - Tried experimenting with different FC layers/orders, but still felt data was insufficient (limited by crawler to 100 images, same as first pull)
- Third pull:
    - Fixed crawler to be able to get >100 images and pulled more images per class
    - Increasing batch size to 64 (previously 32) produced better results

## Character classes
A total of 67 characters can be identified:
- a wing
- admiral ackbar
- anakin skywalker
- atat
- atst
- b wing
- b1 battle droid
- bail organa
- bb8
- boba fett
- c3po
- captain phasma
- chancellor palpatine
- chewbacca
- clone trooper
- count dooku
- darth maul
- darth sidious
- darth vader
- death star
- droideka
- ewok
- finn
- first order stormtrooper
- general grievous
- general hux
- greedo
- han solo
- imperial shuttle
- jabba the hutt
- jango fett
- jar jar binks
- jawa
- ki adi mundi
- kit fisto
- kylo ren
- lando calrissian
- leia organa
- luke skywalker
- mace windu
- millenium falcon
- mon mothma
- nute gunray
- obi wan kenobi
- padme amidala
- plo koon
- poe dameron
- qui gon jinn
- r2d2
- rey
- sand people
- sandcrawler
- shaak ti
- shmi skywalker
- slave 1
- snoke
- snow speeder
- speeder bike
- stardestroyer
- stormtrooper
- super battle droid
- tie bomber
- tie fighter
- wilhuff tarkin
- x wing
- y wing
- yoda