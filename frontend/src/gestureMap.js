export const OPEN_ACTIONS = {
  straight: { type: 'open', modalId: 'welcome', optionId: 'soft' },
  five: { type: 'open', modalId: 'kitchen', optionId: 'silent' },
  peace: { type: 'open', modalId: 'cinema', optionId: 'invisible' },
  okay: { type: 'open', modalId: 'lounge', optionId: 'lounge' },
  rad: { type: 'open', modalId: 'iconic', optionId: 'lounge2' },
  thumbs: { type: 'open', modalId: 'off' },
};

export const NAV_ACTIONS = {
  peace: { type: 'next' },
  okay: { type: 'prev' },
  five: { type: 'confirm' },
  fist: { type: 'close' },
};

export const GESTURE_COOLDOWN_MS = 850;
