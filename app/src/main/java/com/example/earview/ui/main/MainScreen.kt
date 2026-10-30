package com.example.earview.ui.main

import androidx.annotation.StringRes
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.earview.R
import com.example.earview.detection.AndroidDetectionServiceStarter
import com.example.earview.detection.DangerSound
import com.example.earview.detection.DetectionState
import com.example.earview.detectionController
import com.example.earview.permission.AndroidPermissionChecker
import com.example.earview.permission.openAppSettings
import com.example.earview.theme.EarViewTheme
import com.example.earview.theme.StatusDetected
import com.example.earview.theme.StatusIdle
import com.example.earview.theme.StatusOff

@Composable
fun MainScreen(
  modifier: Modifier = Modifier,
  viewModel: MainScreenViewModel = defaultMainScreenViewModel(),
) {
  val context = LocalContext.current
  val state by viewModel.uiState.collectAsStateWithLifecycle()
  val permissionGuideVisible by viewModel.permissionGuideVisible.collectAsStateWithLifecycle()
  MainScreenContent(
    state = state,
    onToggleDetection = viewModel::onToggleDetection,
    onDismissDetection = viewModel::onDismissDetection,
    modifier = modifier,
    permissionGuideVisible = permissionGuideVisible,
    onDismissPermissionGuide = viewModel::onDismissPermissionGuide,
    onOpenSettings = {
      viewModel.onDismissPermissionGuide()
      context.openAppSettings()
    },
  )
}

@Composable
private fun defaultMainScreenViewModel(): MainScreenViewModel {
  val context = LocalContext.current.applicationContext
  return viewModel {
    MainScreenViewModel(context.detectionController, AndroidDetectionServiceStarter(context), AndroidPermissionChecker(context))
  }
}

/** Stateless main screen. Every status is shown with color + icon + text (FR-12). */
@Composable
internal fun MainScreenContent(
  state: DetectionState,
  onToggleDetection: () -> Unit,
  onDismissDetection: () -> Unit,
  modifier: Modifier = Modifier,
  permissionGuideVisible: Boolean = false,
  onDismissPermissionGuide: () -> Unit = {},
  onOpenSettings: () -> Unit = {},
) {
  val background = state.backgroundColor()
  val dismissLabel = stringResource(R.string.action_dismiss_detection)
  val dismissModifier =
    if (state is DetectionState.Detected) {
      Modifier.clickable(onClickLabel = dismissLabel, onClick = onDismissDetection)
    } else {
      Modifier
    }

  Box(modifier.fillMaxSize().background(background).then(dismissModifier)) {
    Column(
      modifier = Modifier.fillMaxSize().safeDrawingPadding().padding(16.dp),
      horizontalAlignment = Alignment.CenterHorizontally,
    ) {
      StatusHeader(state)
      Box(modifier = Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
        ToggleButton(isRunning = state != DetectionState.Off, contentColor = background, onClick = onToggleDetection)
      }
      TargetSoundList()
    }
  }

  if (permissionGuideVisible) {
    PermissionGuideDialog(onOpenSettings = onOpenSettings, onDismiss = onDismissPermissionGuide)
  }
}

/** Explains why detection did not start and offers a way to the settings screen (BL-022, BL-023). */
@Composable
private fun PermissionGuideDialog(onOpenSettings: () -> Unit, onDismiss: () -> Unit) {
  AlertDialog(
    onDismissRequest = onDismiss,
    icon = { Icon(imageVector = Icons.Filled.Warning, contentDescription = null) },
    title = { Text(stringResource(R.string.permission_guide_title)) },
    text = { Text(stringResource(R.string.permission_guide_message)) },
    confirmButton = { TextButton(onClick = onOpenSettings) { Text(stringResource(R.string.permission_guide_open_settings)) } },
    dismissButton = { TextButton(onClick = onDismiss) { Text(stringResource(R.string.permission_guide_close)) } },
  )
}

@Composable
private fun StatusHeader(state: DetectionState, modifier: Modifier = Modifier) {
  val text = state.statusText()
  val spoken = state.spokenText()
  Row(
    modifier = modifier.fillMaxWidth().semantics(mergeDescendants = true) {
      contentDescription = spoken
      liveRegion = LiveRegionMode.Polite
    },
    verticalAlignment = Alignment.CenterVertically,
    horizontalArrangement = Arrangement.Center,
  ) {
    Icon(imageVector = state.icon(), contentDescription = null, tint = Color.White, modifier = Modifier.size(48.dp))
    Spacer(Modifier.width(12.dp))
    Text(
      text = text,
      style = MaterialTheme.typography.headlineMedium,
      fontWeight = FontWeight.Bold,
      color = Color.White,
      textAlign = TextAlign.Center,
      modifier = Modifier.weight(1f, fill = false),
    )
  }
}

@Composable
private fun ToggleButton(isRunning: Boolean, contentColor: Color, onClick: () -> Unit) {
  val label = stringResource(if (isRunning) R.string.button_stop else R.string.button_start)
  val spoken = stringResource(if (isRunning) R.string.button_stop_a11y else R.string.button_start_a11y)
  Button(
    onClick = onClick,
    shape = CircleShape,
    colors = ButtonDefaults.buttonColors(containerColor = Color.White, contentColor = contentColor),
    modifier = Modifier.size(200.dp).semantics { contentDescription = spoken },
  ) {
    Text(text = label, style = MaterialTheme.typography.headlineLarge, fontWeight = FontWeight.Bold)
  }
}

@Composable
private fun TargetSoundList(modifier: Modifier = Modifier) {
  Text(
    text = stringResource(R.string.target_sounds, DangerSound.entries.joinedNames()),
    style = MaterialTheme.typography.bodyLarge,
    color = Color.White,
    textAlign = TextAlign.Center,
    modifier = modifier.fillMaxWidth(),
  )
}

private fun DetectionState.backgroundColor(): Color = when (this) {
  DetectionState.Off -> StatusOff
  DetectionState.Idle -> StatusIdle
  is DetectionState.Detected -> StatusDetected
}

private fun DetectionState.icon(): ImageVector = when (this) {
  DetectionState.Off -> Icons.Filled.Close
  DetectionState.Idle -> Icons.Filled.CheckCircle
  is DetectionState.Detected -> Icons.Filled.Warning
}

@Composable
private fun DetectionState.statusText(): String = when (this) {
  DetectionState.Off -> stringResource(R.string.status_off)
  DetectionState.Idle -> stringResource(R.string.status_idle)
  is DetectionState.Detected ->
    detectedText(R.string.status_detected, R.string.status_detected_overflow, R.string.status_detected_generic)
}

@Composable
private fun DetectionState.spokenText(): String = when (this) {
  DetectionState.Off -> stringResource(R.string.status_off_a11y)
  DetectionState.Idle -> stringResource(R.string.status_idle_a11y)
  is DetectionState.Detected ->
    detectedText(R.string.status_detected_a11y, R.string.status_detected_overflow_a11y, R.string.status_detected_generic_a11y)
}

@Composable
private fun DetectionState.Detected.detectedText(
  @StringRes listRes: Int,
  @StringRes overflowRes: Int,
  @StringRes genericRes: Int,
): String {
  if (sounds.isEmpty()) return stringResource(genericRes)
  val names = sounds.joinedNames()
  return if (overflowCount > 0) stringResource(overflowRes, names, overflowCount) else stringResource(listRes, names)
}

@Composable
private fun List<DangerSound>.joinedNames(): String {
  val separator = stringResource(R.string.sound_separator)
  return map { stringResource(it.labelRes()) }.joinToString(separator)
}

@StringRes
private fun DangerSound.labelRes(): Int = when (this) {
  DangerSound.FIRE_ALARM -> R.string.sound_fire_alarm
  DangerSound.SIREN -> R.string.sound_siren
  DangerSound.GLASS_BREAKING -> R.string.sound_glass_breaking
  DangerSound.DOORBELL -> R.string.sound_doorbell
  DangerSound.DOOR_KNOCK -> R.string.sound_door_knock
  DangerSound.CAR_HORN -> R.string.sound_car_horn
}

@Preview(showBackground = true)
@Composable
private fun MainScreenOffPreview() {
  EarViewTheme { MainScreenContent(DetectionState.Off, onToggleDetection = {}, onDismissDetection = {}) }
}

@Preview(showBackground = true)
@Composable
private fun MainScreenIdlePreview() {
  EarViewTheme { MainScreenContent(DetectionState.Idle, onToggleDetection = {}, onDismissDetection = {}) }
}

@Preview(showBackground = true)
@Composable
private fun MainScreenDetectedPreview() {
  EarViewTheme {
    MainScreenContent(
      DetectionState.Detected(listOf(DangerSound.FIRE_ALARM, DangerSound.SIREN, DangerSound.GLASS_BREAKING), overflowCount = 3),
      onToggleDetection = {},
      onDismissDetection = {},
    )
  }
}
