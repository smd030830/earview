package com.example.earview.detection

/**
 * The six danger sounds the app detects (fixed for the MVP).
 *
 * Declaration order is the display priority (most dangerous first), so sorting by natural
 * order yields the order required by FR-04. The order is a proposal pending user testing.
 */
enum class DangerSound {
  FIRE_ALARM,
  SIREN,
  GLASS_BREAKING,
  DOORBELL,
  DOOR_KNOCK,
  CAR_HORN,
}
