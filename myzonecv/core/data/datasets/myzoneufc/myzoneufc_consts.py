from myzonecv.core import consts as C

NONFIGHTER_LABEL = 'non-fighter'

REID_BBOX_PADDING_RATIO = 1
REID_BBOX_SCALE_UNIT = C.BBOX_SCALE_UNIT
REID_BORDER_COLOR_VALUE = C.BORDER_COLOR_VALUE

CAT_NAME_FIGHTER_LEFT = 'fighter-left'
CAT_ID_FIGHTER_LEFT = 0
CAT_NAME_FIGHTER_RIGHT = 'fighter-right'
CAT_ID_FIGHTER_RIGHT = 1
CAT_NAME_FIGHTER_NEAR = 'fighter-near'
CAT_ID_FIGHTER_NEAR = 2
CAT_NAME_FIGHTER_FAR = 'fighter-far'
CAT_ID_FIGHTER_FAR = 3
CAT_NAME_FIGHTER_UP = 'fighter-up'
CAT_ID_FIGHTER_UP = 4
CAT_NAME_FIGHTER_DOWN = 'fighter-down'
CAT_ID_FIGHTER_DOWN = 5
CAT_NAME_FIGHTER_CROWD = 'fighter-crowd'
CAT_ID_FIGHTER_CROWD = 6
CAT_NAME_FIGHTER_SINGLE = 'fighter-single'
CAT_ID_FIGHTER_SINGLE = 7
CAT_NAME_FIGHTER_NONE = 'fighter-none'
CAT_ID_FIGHTER_NONE = 8

CAT_ID_FIGHTER_LEFTRIGHT = 8  # overwrite 'fighter-none'
CAT_ID_FIGHTER_NEARFAR = 9
CAT_ID_FIGHTER_UPDOWN = 10
