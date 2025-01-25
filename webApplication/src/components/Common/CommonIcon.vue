<script>
import {defineComponent} from 'vue'

export default defineComponent({
    name: "CommonIcon",
    props: {
        iconFile: String,
        name: String,
    },
    computed:{
        iconPath() {
            return new URL("/src/assets/icons/" + this.iconFile, import.meta.url);
        }
    },
})
</script>

<template>
    <div>
        <img :alt="name + ' icon'" :src="iconPath">
    </div>
</template>

<style scoped>

</style>