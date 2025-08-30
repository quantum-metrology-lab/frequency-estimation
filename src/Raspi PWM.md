# Hardware PWM on the Raspberry Pi 5 b

> *Author: Gadgetoid*
>
> *Source: [Raspberry Pi 5 - All channels on pwm0](https://gist.github.com/Gadgetoid/b92ad3db06ff8c264eef2abf0e09d569)*
>
> *Forked @ 2025.3.25 and modified by Chao-Ning Hu*

Since PWM is a little fraught with gotchas, this is mostly a message to future me-

(Note to self, rtfm - https://datasheets.raspberrypi.com/rp1/rp1-peripherals.pdf)

pin    | a0         | a3         |
-------|------------|------------|
GPIO19 |            | PWM0_CHAN3 |
GPIO18 |            | PWM0_CHAN2 |
GPIO15 | PWM0_CHAN3 |            |
GPIO14 | PWM0_CHAN2 |            |
GPIO13 | PWM0_CHAN1 |            |
GPIO12 | PWM0_CHAN0 |            |

TODO: Figure out how to tell if pwm0 is on `/sys/class/pwm/pwmchip1` or  `/sys/class/pwm/pwmchip2`. pwm1 on the Pi 5 might have 
`device/consumer:platform:cooling_fan/`

Life is short, this single dtoverlay configures GPIO12, GPIO13, GPIO18 and GPIO19 to their respective alt modes on boot and enables pwm0:

```dts
/dts-v1/;
/plugin/;

/{
	compatible = "brcm,bcm2712";

	fragment@0 {
		target = <&rp1_gpio>;
		__overlay__ {
			pwm_pins: pwm_pins {
				pins = "gpio12", "gpio13", "gpio18", "gpio19";
				function = "pwm0", "pwm0", "pwm0", "pwm0";
			};
		};
	};

	fragment@1 {
		target = <&rp1_pwm0>;
		frag1: __overlay__ {
			pinctrl-names = "default";
			pinctrl-0 = <&pwm_pins>;
			status = "okay";
		};
	};
};
```

Save as "pwm-pi5-overlay.dts" and compile with:

```
dtc -I dts -O dtb -o pwm-pi5.dtbo pwm-pi5-overlay.dts
```

Install:

```
sudo cp pwm-pi5.dtbo /boot/firmware/overlays/
```

Don't forget to add `dtoverlay=pwm-pi5` to `/boot/firmware/config.txt`...

Then use this janky script to stick some safety rails on poking PWM:

```bash
#!/bin/bash
NODE=/sys/class/pwm/pwmchip2  # by cnhu
CHANNEL="$1"
PERIOD="$2"
DUTY_CYCLE="$3"

function usage {
	printf "Usage: $0 <channel> <period> <duty_cycle>\n"
	printf "    channel - number from 0-3\n"
	printf "    period - PWM period in nanoseconds\n"
	printf "    duty_cycle - Duty Cycle (on period) in nanoseconds\n"
	exit 1
}

if [[ ! $CHANNEL =~ ^[0-3]+$ ]]; then
	usage
fi

if [ -d "$NODE/device/consumer:platform:cooling_fan/" ]; then
	echo "Hold your horses, looks like this is pwm1?"
	exit 1
fi

if [ ! -d "$NODE/pwm$CHANNEL" ]; then
	echo $CHANNEL | sudo tee -a "$NODE/export"  # by cnhu
fi

echo "0" | sudo tee -a "$NODE/pwm$CHANNEL/enable" > /dev/null
echo "$PERIOD" | sudo tee -a "$NODE/pwm$CHANNEL/period" > /dev/null
if [ $? -ne 0 ]; then
	echo "^ don't worry, handling it!"
	echo "$DUTY_CYCLE" | sudo tee -a "$NODE/pwm$CHANNEL/duty_cycle" > /dev/null
	echo "$PERIOD" | sudo tee -a "$NODE/pwm$CHANNEL/period" > /dev/null
else
	echo "$DUTY_CYCLE" | sudo tee -a "$NODE/pwm$CHANNEL/duty_cycle" > /dev/null
fi
echo "1" | sudo tee -a "$NODE/pwm$CHANNEL/enable" > /dev/null


case $CHANNEL in
	"0")
	PIN="12"
	FUNC="a0"
	;;
	"1")
	PIN="13"
	FUNC="a0"
	;;
	"2")
	PIN="18"
	FUNC="a3"
	;;
	"3")
	PIN="19"
	FUNC="a3"
esac

# Sure, the pin is set to the correct alt mode by the dtoverlay at startup...
# But we'll do this to protect the user (me, the user is me) from themselves:
pinctrl set $PIN $FUNC

echo "PWM$CHANNEL set to $PERIOD ns, $DUTY_CYCLE, on pin $PIN (func $FUNC)."
```

