ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_PYLON = 'buildpylon'
ACTION_BUILD_GATEWAY = 'buildgateway'
ACTION_BUILD_CYBERNETICSCORE = 'buildcyberneticscore'
ACTION_BUILD_ASSIMILATOR = 'buildassimilator'
ACTION_HARVEST_GAS = 'harvestgas'
ACTION_BUILD_ZEALOT = 'buildzealot'
ACTION_BUILD_STALKER = 'buildstalker'
ACTION_BUILD_ADEPT = 'buildadept'
ACTION_BUILD_SENTRY = 'buildsentry'
ACTION_ATTACK = 'attack'

ai_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_PYLON,
    ACTION_BUILD_GATEWAY,
    ACTION_BUILD_CYBERNETICSCORE,
    ACTION_BUILD_ASSIMILATOR,
    ACTION_HARVEST_GAS,
    ACTION_BUILD_ZEALOT,
    ACTION_BUILD_STALKER,
    ACTION_BUILD_ADEPT,
    ACTION_BUILD_SENTRY,
]

# import logging
for mm_x in range(0, 4):
    for mm_y in range(0, 4):
        # logging.info(ACTION_ATTACK + '_' + str((mm_x * 16) + 8) + '_' + str((mm_y * 16) + 8))
        ai_actions.append(ACTION_ATTACK + '_' + str((mm_x * 16) + 8) + '_' + str((mm_y * 16) + 8))