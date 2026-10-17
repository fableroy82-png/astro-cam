#include <Arduino.h>
#include <Arduino_RouterBridge.h>

// Pont ST4 pour Arduino UNO Q.
// IMPORTANT : ces broches doivent attaquer des optocoupleurs, jamais le port
// ST4 de l'EQ6 directement. Vérifier au multimètre que les quatre sorties sont
// ouvertes au repos avant de connecter la monture.

static constexpr uint8_t PIN_N = D2;
static constexpr uint8_t PIN_S = D3;
static constexpr uint8_t PIN_E = D4;
static constexpr uint8_t PIN_W = D5;

// Montage prévu : HIGH allume la LED de l'optocoupleur, LOW = repos.
// Inverser ces deux constantes si votre carte optocouplée est active à LOW.
static constexpr uint8_t ST4_ACTIVE_LEVEL = HIGH;
static constexpr uint8_t ST4_IDLE_LEVEL = LOW;
static constexpr int MAX_PULSE_MS = 1500;

volatile bool pulseActive = false;
volatile uint8_t activePin = 255;
volatile unsigned long pulseEndsAt = 0;

void allOff() {
  digitalWrite(PIN_N, ST4_IDLE_LEVEL);
  digitalWrite(PIN_S, ST4_IDLE_LEVEL);
  digitalWrite(PIN_E, ST4_IDLE_LEVEL);
  digitalWrite(PIN_W, ST4_IDLE_LEVEL);
  pulseActive = false;
  activePin = 255;
}

uint8_t pinForDirection(const String &direction) {
  if (direction == "N") return PIN_N;
  if (direction == "S") return PIN_S;
  if (direction == "E") return PIN_E;
  if (direction == "W") return PIN_W;
  return 255;
}

bool st4_pulse(String direction, int durationMs) {
  direction.toUpperCase();
  const uint8_t pin = pinForDirection(direction);
  if (pin == 255 || durationMs <= 0 || durationMs > MAX_PULSE_MS) {
    allOff();
    return false;
  }

  // Une seule direction à la fois : évite toute combinaison N+S ou E+W.
  if (pulseActive) return false;

  allOff();
  digitalWrite(pin, ST4_ACTIVE_LEVEL);
  activePin = pin;
  pulseEndsAt = millis() + static_cast<unsigned long>(durationMs);
  pulseActive = true;
  return true;
}

void setup() {
  pinMode(PIN_N, OUTPUT);
  pinMode(PIN_S, OUTPUT);
  pinMode(PIN_E, OUTPUT);
  pinMode(PIN_W, OUTPUT);
  allOff();

  Bridge.begin();
  Bridge.provide_safe("st4_pulse", st4_pulse);
}

void loop() {
  if (pulseActive && static_cast<long>(millis() - pulseEndsAt) >= 0) {
    allOff();
  }
  delay(1);
}

