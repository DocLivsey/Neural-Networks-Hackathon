<script>
import {defineComponent} from 'vue'
import CommonIcon from "@/components/Common/CommonIcon.vue";

export default defineComponent({
    name: "LoadingBlinker",
    components: {CommonIcon}
})
</script>

<template>
    <CommonIcon class="blinker" icon-file="filled-dot.svg"/>
</template>

<style scoped>
.blinker {
    animation: 1s infinite alternate blinker;
}
@keyframes blinker {
    0% {
        opacity: 0;
    }
    50% {
        opacity: 1;
    }
    100% {
        opacity: 0;
    }
}
</style>