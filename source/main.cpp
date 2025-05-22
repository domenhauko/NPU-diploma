/*
 * Copyright 2020-2022 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "board_init.h"
#include "demo_config.h"
#include "demo_info.h"
#include "fsl_debug_console.h"
#include "image.h"
#include "image_utils.h"
#include "model.h"
#include "output_postproc.h"
#include "timer.h"

#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"

static SemaphoreHandle_t xEventSemaphore = NULL;

void MobileNetTask1(void *pvParameters)
{
    PRINTF("Task started\r\n");

    if (MODEL_Init() != kStatus_Success)
    {
        PRINTF("Model init failed\r\n");
        vTaskSuspend(NULL);
    }
    PRINTF("Model initialized\r\n");

    tensor_dims_t inputDims;
    tensor_type_t inputType;
    uint8_t* inputData = MODEL_GetInputTensorData(&inputDims, &inputType);
    PRINTF("Got input tensor\r\n");

    tensor_dims_t outputDims;
    tensor_type_t outputType;
    uint8_t* outputData = MODEL_GetOutputTensorData(&outputDims, &outputType);
    PRINTF("Got output tensor\r\n");

    while (1)
    {
        PRINTF("Getting image...\r\n");
        if (IMAGE_GetImage(inputData, inputDims.data[2], inputDims.data[1], inputDims.data[3]) != kStatus_Success)
        {
            PRINTF("IMAGE FAILED\r\n");
            continue;
        }

        PRINTF("Image acquired. Running inference...\r\n");
        MODEL_ConvertInput(inputData, &inputDims, inputType);

        auto startTime = TIMER_GetTimeInUS();
        MODEL_RunInference();
        auto endTime = TIMER_GetTimeInUS();

        PRINTF("Inference done. Processing output...\r\n");
        MODEL_ProcessOutput(outputData, &outputDims, outputType, endTime - startTime);

        PRINTF("Free heap: %u bytes\r\n", xPortGetFreeHeapSize());
        PRINTF("Stack usage (remaining): %u words\r\n", uxTaskGetStackHighWaterMark(NULL));
        vTaskDelay(pdMS_TO_TICKS(1000));


    }
}

void RealTimeTask(void *pvParameters)
{
    while (1)
    {
        if (xSemaphoreTake(xEventSemaphore, portMAX_DELAY) == pdTRUE)
        {
            PRINTF("[RealTimeTask] Semaphore received! Executing RT task.\r\n");
        }
    }
}

extern "C" void vApplicationTickHook(void)
{
    static uint32_t tickCount = 0;
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;

    tickCount++;
    if (tickCount >= 500) // 500 ticks = 2.5 seconds if tick = 5ms
    {
        xSemaphoreGiveFromISR(xEventSemaphore, &xHigherPriorityTaskWoken);
        tickCount = 0;
    }
}

extern "C" void vApplicationMallocFailedHook(void)
{
    PRINTF("Malloc failed! Out of FreeRTOS heap.\r\n");
    taskDISABLE_INTERRUPTS();
    for (;;) {}
}

extern "C" void vApplicationStackOverflowHook(TaskHandle_t xTask, char *pcTaskName)
{
    PRINTF("Stack overflow in task: %s\r\n", pcTaskName);
    taskDISABLE_INTERRUPTS();
    for (;;) {}
}

int main(void)
{
     BOARD_Init();
    TIMER_Init();
    DEMO_PrintInfo();

    xEventSemaphore = xSemaphoreCreateBinary();

    xTaskCreate(MobileNetTask1, "ModelA", 8192, NULL, 2, NULL);
    xTaskCreate(RealTimeTask, "RT_Task", 512, NULL, configMAX_PRIORITIES - 1, NULL);



    vTaskStartScheduler();



    for (;;) {} // Should never reach here
}

//int main(void)
//{
//    BOARD_Init();
//    TIMER_Init();
//
//    DEMO_PrintInfo();
//
//    if (MODEL_Init() != kStatus_Success)
//    {
//        PRINTF("Failed initializing model" EOL);
//        for (;;) {}
//    }
//
//    tensor_dims_t inputDims;
//    tensor_type_t inputType;
//    uint8_t* inputData = MODEL_GetInputTensorData(&inputDims, &inputType);
//
//    tensor_dims_t outputDims;
//    tensor_type_t outputType;
//    uint8_t* outputData = MODEL_GetOutputTensorData(&outputDims, &outputType);
//
//    while (1)
//    {
//        /* Expected tensor dimensions: [batches, height, width, channels] */
//        if (IMAGE_GetImage(inputData, inputDims.data[2], inputDims.data[1], inputDims.data[3]) != kStatus_Success)
//        {
//            PRINTF("Failed retrieving input image" EOL);
//            for (;;) {}
//        }
//
//        MODEL_ConvertInput(inputData, &inputDims, inputType);
//
//        auto startTime = TIMER_GetTimeInUS();
//        MODEL_RunInference();
//        auto endTime = TIMER_GetTimeInUS();
//
//        MODEL_ProcessOutput(outputData, &outputDims, outputType, endTime - startTime);
//    }
//}
