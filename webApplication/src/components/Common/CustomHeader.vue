<script>
import {defineComponent} from 'vue'
import CommonIcon from "@/components/Common/CommonIcon.vue";

export default defineComponent({
    name: "CustomHeader",
    props: {
        text: String,
        routeTo: String,
    },
    components: {CommonIcon}
})
</script>

<template>
    <div class="header">
        <RouterLink :to="`${routeTo}`"><CommonIcon icon-file="arrow.svg"/></RouterLink>
        <div class="header__text">
            {{text}}
        </div>
        <div class="header__balancer">

        </div>
    </div>
</template>

<style scoped>
.header {
    display: flex;
    flex-direction: row;
    align-items: center;
    padding-top: 0.9375em;
    padding-left: 0.9375em;
    padding-right: 0.9375em;
    justify-content: space-between;
}
.header__text {
    font-size: var(--secondary-font-size);
    font-family: var(--secondary-font-family);
    font-weight: 600;
}
.header__balancer {
    width: 2.25em;
}
</style>