################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include utilities/str/subdir.mk
-include utilities/debug_console_lite/subdir.mk
-include utilities/subdir.mk
-include startup/subdir.mk
-include source/subdir.mk
-include freertos/freertos-kernel/portable/MemMang/subdir.mk
-include freertos/freertos-kernel/portable/GCC/ARM_CM33_NTZ/non_secure/subdir.mk
-include freertos/freertos-kernel/subdir.mk
-include freertos/corejson/source/subdir.mk
-include eiq/tensorflow-lite/tensorflow/lite/micro/kernels/neutron/subdir.mk
-include eiq/tensorflow-lite/tensorflow/lite/micro/subdir.mk
-include drivers/subdir.mk
-include device/subdir.mk
-include component/uart/subdir.mk
-include component/lists/subdir.mk
-include board/subdir.mk
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C++M_DEPS)),)
-include $(C++M_DEPS)
endif
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(CCM_DEPS)),)
-include $(CCM_DEPS)
endif
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
ifneq ($(strip $(CXXM_DEPS)),)
-include $(CXXM_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
endif

-include ../makefile.defs

OPTIONAL_TOOL_DEPS := \
$(wildcard ../makefile.defs) \
$(wildcard ../makefile.init) \
$(wildcard ../makefile.targets) \


BUILD_ARTIFACT_NAME := frdmmcxn947_tflm_label_image_cm33_core0
BUILD_ARTIFACT_EXTENSION := axf
BUILD_ARTIFACT_PREFIX :=
BUILD_ARTIFACT := $(BUILD_ARTIFACT_PREFIX)$(BUILD_ARTIFACT_NAME)$(if $(BUILD_ARTIFACT_EXTENSION),.$(BUILD_ARTIFACT_EXTENSION),)

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all:
	+@$(MAKE) --no-print-directory main-build && $(MAKE) --no-print-directory post-build

# Main-build Target
main-build: frdmmcxn947_tflm_label_image_cm33_core0.axf

# Tool invocations
frdmmcxn947_tflm_label_image_cm33_core0.axf: $(OBJS) $(USER_OBJS) makefile $(OPTIONAL_TOOL_DEPS)
	@echo 'Building target: $@'
	@echo 'Invoking: MCU C++ Linker'
	arm-none-eabi-c++ -nostdlib -L"C:\Users\Domen\Documents\MULTIMEDIJA\Diploma\Implementation\FreeRTOS_implementation_Mobilenet_Project_Gorkem_Salman\frdmmcxn947_tflm_label_image_cm33_core0\eiq\tensorflow-lite\lib\cm33\armgcc" -L"C:\Users\Domen\Documents\MULTIMEDIJA\Diploma\Implementation\FreeRTOS_implementation_Mobilenet_Project_Gorkem_Salman\frdmmcxn947_tflm_label_image_cm33_core0\eiq\tensorflow-lite\third_party\neutron\mcxn" -Xlinker -no-warn-rwx-segments -Xlinker -Map="frdmmcxn947_tflm_label_image_cm33_core0.map" -Xlinker --gc-sections -Xlinker -print-memory-usage -Xlinker --sort-section=alignment -Xlinker --cref -mcpu=cortex-m33 -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -T frdmmcxn947_tflm_label_image_cm33_core0_Debug.ld -o "frdmmcxn947_tflm_label_image_cm33_core0.axf" $(OBJS) $(USER_OBJS) $(LIBS) -ltflm -lNeutronDriver -lNeutronFirmware
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) frdmmcxn947_tflm_label_image_cm33_core0.axf
	-@echo ' '

post-build:
	-@echo 'Performing post-build steps'
	-arm-none-eabi-size "frdmmcxn947_tflm_label_image_cm33_core0.axf"; # arm-none-eabi-objcopy -v -O binary "frdmmcxn947_tflm_label_image_cm33_core0.axf" "frdmmcxn947_tflm_label_image_cm33_core0.bin" ; # checksum -p MCXN947 -d "frdmmcxn947_tflm_label_image_cm33_core0.bin";
	-@echo ' '

.PHONY: all clean dependents main-build post-build

-include ../makefile.targets
