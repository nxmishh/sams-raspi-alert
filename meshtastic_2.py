import time
import meshtastic
import meshtastic.serial_interface
from pubsub import pub
from threading import Thread
import pygame

# Path to your buzzer sound file
buzzer_sound_file = "alert.mp3"

def on_receive(packet, interface):
    print(packet['decoded']['payload'])
    try:
        if packet['decoded']['payload'] == b"ALERT! Yawning":
            play_sound(buzzer_sound_file)
        elif packet['decoded']['payload'] == b"Alert!! Eyes closed":
            play_sound(buzzer_sound_file)
    except:
        print("No new packet")
    

def on_connection(interface, topic=pub.AUTO_TOPIC):
    # Send a message when connected
    interface.sendText("hello mesh")

def play_sound(sound_file):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Error playing sound: {e}")

def main():
    try:
        # Initialize Meshtastic interface
        interface = meshtastic.serial_interface.SerialInterface()

        # Subscribe to the receive message topic
        pub.subscribe(on_receive, "meshtastic.receive")

        # Subscribe to the connection topic

        # Keep the script running to listen for messages
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting script.")
    finally:
        interface.close()

if __name__ == "__main__":
    main()
