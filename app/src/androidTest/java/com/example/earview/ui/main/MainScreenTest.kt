package com.example.earview.ui.main

import androidx.activity.ComponentActivity
import androidx.compose.ui.test.hasContentDescription
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import com.example.earview.detection.DangerSound
import com.example.earview.detection.DetectionState
import com.example.earview.theme.EarViewTheme
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test

/** UI tests for [MainScreenContent]. */
class MainScreenTest {
  @get:Rule val composeTestRule = createAndroidComposeRule<ComponentActivity>()

  private fun show(state: DetectionState, onToggle: () -> Unit = {}, onDismiss: () -> Unit = {}) {
    composeTestRule.setContent {
      EarViewTheme { MainScreenContent(state, onToggleDetection = onToggle, onDismissDetection = onDismiss) }
    }
  }

  @Test
  fun off_showsStatusText_andStartButton() {
    show(DetectionState.Off)

    composeTestRule.onNodeWithText("꺼져 있음").assertExists()
    composeTestRule.onNodeWithText("시작").assertExists()
    composeTestRule.onNodeWithContentDescription("소리 감지 시작").assertExists()
  }

  @Test
  fun idle_showsStatusText_andStopButton() {
    show(DetectionState.Idle)

    composeTestRule.onNodeWithText("정상 대기 중").assertExists()
    composeTestRule.onNodeWithText("종료").assertExists()
    composeTestRule.onNodeWithContentDescription("소리 감지 종료").assertExists()
  }

  @Test
  fun detected_showsSoundNames() {
    show(DetectionState.Detected(listOf(DangerSound.FIRE_ALARM, DangerSound.SIREN), overflowCount = 0))

    composeTestRule.onNodeWithText("화재경보, 사이렌 감지").assertExists()
    composeTestRule.onNodeWithText("종료").assertExists()
  }

  @Test
  fun detected_withOverflow_showsOverflowCount() {
    show(
      DetectionState.Detected(
        listOf(DangerSound.FIRE_ALARM, DangerSound.SIREN, DangerSound.GLASS_BREAKING),
        overflowCount = 3,
      ),
    )

    composeTestRule.onNodeWithText("화재경보, 사이렌, 유리 깨짐 외 3건 감지").assertExists()
  }

  @Test
  fun detected_isReadByTalkBackAsOneSentence() {
    show(DetectionState.Detected(listOf(DangerSound.FIRE_ALARM, DangerSound.SIREN), overflowCount = 0))

    composeTestRule.onNode(hasContentDescription("화재경보, 사이렌 감지됨")).assertExists()
  }

  @Test
  fun toggleButton_invokesCallback() {
    var toggles = 0
    show(DetectionState.Off, onToggle = { toggles++ })

    composeTestRule.onNodeWithText("시작").performClick()

    assertEquals(1, toggles)
  }

  @Test
  fun tapOnScreenWhileDetected_invokesDismiss() {
    var dismissals = 0
    show(DetectionState.Detected(listOf(DangerSound.SIREN), overflowCount = 0), onDismiss = { dismissals++ })

    composeTestRule.onNodeWithText("사이렌 감지").performClick()

    assertEquals(1, dismissals)
  }

  @Test
  fun listsAllTargetSounds() {
    show(DetectionState.Idle)

    composeTestRule.onNodeWithText("감지 대상: 화재경보, 사이렌, 유리 깨짐, 초인종, 문 두드림, 차량 경적").assertExists()
    composeTestRule.onNode(hasText("아기 울음", substring = true)).assertDoesNotExist()
  }

  @Test
  fun newSounds_areDisplayedAndReadByTalkBack() {
    show(DetectionState.Detected(listOf(DangerSound.GLASS_BREAKING, DangerSound.DOOR_KNOCK), overflowCount = 0))

    composeTestRule.onNodeWithText("유리 깨짐, 문 두드림 감지").assertExists()
    composeTestRule.onNodeWithContentDescription("유리 깨짐, 문 두드림 감지됨").assertExists()
  }

  @Test
  fun permissionGuide_isShown_withSettingsButton() {
    var opened = 0
    composeTestRule.setContent {
      EarViewTheme {
        MainScreenContent(
          DetectionState.Off,
          onToggleDetection = {},
          onDismissDetection = {},
          permissionGuideVisible = true,
          onOpenSettings = { opened++ },
        )
      }
    }

    composeTestRule.onNodeWithText("마이크 권한이 필요합니다").assertExists()
    composeTestRule.onNodeWithText("설정 열기").performClick()

    assertEquals(1, opened)
  }

  @Test
  fun permissionGuide_isHidden_byDefault() {
    show(DetectionState.Off)

    composeTestRule.onNodeWithText("마이크 권한이 필요합니다").assertDoesNotExist()
  }
}
